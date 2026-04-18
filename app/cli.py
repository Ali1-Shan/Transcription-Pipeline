"""CLI interface for the transcription pipeline.

Allows running transcription from the command line without starting
the API server. Useful for batch processing and scripting.

Usage:
    python -m app.cli transcribe audio.wav
    python -m app.cli transcribe audio.mp3 --model small --output result.json
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

from loguru import logger
from pydub import AudioSegment

from app.config import Settings, VALID_WHISPER_MODELS
from app.core.formatter import OutputFormatter
from app.core.postprocessor import PostProcessor
from app.core.transcriber import Transcriber
from app.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="transcription-pipeline",
        description="Audio transcription pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- transcribe command ---
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe an audio file",
    )
    transcribe_parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file (WAV or MP3)",
    )
    transcribe_parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(VALID_WHISPER_MODELS),
        help="Whisper model size (overrides .env setting)",
    )
    transcribe_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (prints to stdout if not specified)",
    )
    transcribe_parser.add_argument(
        "--no-fillers",
        action="store_true",
        default=False,
        help="Disable filler word removal",
    )
    transcribe_parser.add_argument(
        "--no-punctuation",
        action="store_true",
        default=False,
        help="Disable punctuation correction",
    )
    transcribe_parser.add_argument(
        "--no-segmentation",
        action="store_true",
        default=False,
        help="Disable speaker segmentation",
    )

    return parser.parse_args()


async def run_transcription(args: argparse.Namespace) -> None:
    """Execute the transcription pipeline on a local file.

    Args:
        args: Parsed CLI arguments.
    """
    audio_path = Path(args.audio_file)

    if not audio_path.exists():
        logger.error("File not found: {}", audio_path)
        sys.exit(1)

    if audio_path.suffix.lower() not in {".wav", ".mp3"}:
        logger.error("Unsupported file format: {}. Use WAV or MP3.", audio_path.suffix)
        sys.exit(1)

    # Build settings with CLI overrides
    overrides: dict = {}
    if args.model:
        overrides["whisper_model_size"] = args.model
    if args.no_fillers:
        overrides["enable_filler_removal"] = False
    if args.no_punctuation:
        overrides["enable_punctuation_correction"] = False
    if args.no_segmentation:
        overrides["enable_speaker_segmentation"] = False

    settings = Settings(**overrides)

    # Initialize pipeline
    transcriber = Transcriber(settings)
    post_processor = PostProcessor(settings)
    formatter = OutputFormatter()

    # Get audio duration via pydub
    try:
        if audio_path.suffix.lower() == ".mp3":
            audio = AudioSegment.from_mp3(str(audio_path))
        else:
            audio = AudioSegment.from_wav(str(audio_path))
    except Exception as exc:
        logger.error("Failed to read audio file: {}", str(exc))
        sys.exit(1)
    duration = len(audio) / 1000.0

    # Run pipeline
    start_time = time.monotonic()

    try:
        logger.info("Transcribing: {} ({:.1f}s)", audio_path.name, duration)
        result = await transcriber.transcribe(audio_path)

        cleaned_text, segments = post_processor.process(
            text=result.text,
            word_timestamps=result.word_timestamps,
            duration=duration,
            language=result.language,
        )

        processing_time = time.monotonic() - start_time

        response = formatter.format(
            transcript=cleaned_text,
            confidence=result.confidence,
            language=result.language,
            processing_time=processing_time,
            word_timestamps=result.word_timestamps,
            segments=segments,
            filename=audio_path.name,
            duration=duration,
            model_used=transcriber.model_name,
        )
    except Exception as exc:
        logger.error("Pipeline failed: {}", str(exc))
        sys.exit(1)

    # Output
    json_output = response.model_dump_json(indent=2)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json_output, encoding="utf-8")
        logger.info("Result written to: {}", output_path)
    else:
        print(json_output)

    logger.info(
        "Done | duration={:.1f}s | processing={:.2f}s | language={}",
        duration,
        processing_time,
        result.language,
    )


def main() -> None:
    """CLI entry point."""
    settings = Settings()
    setup_logger(settings.log_level)
    args = parse_args()

    if args.command == "transcribe":
        asyncio.run(run_transcription(args))
    else:
        logger.error("Unknown command. Use 'transcribe'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
