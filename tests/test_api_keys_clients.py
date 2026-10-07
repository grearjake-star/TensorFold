"""The serve command takes repeated API keys, a key file and the metrics exception."""

from tensorfold.cli import build_parser


def test_serve_accepts_repeated_keys_and_metrics_exception():
    args = build_parser().parse_args(["serve", "fixture", "--api-key", "fixture-first", "--api-key", "fixture-second",
                                     "--api-key-file", "keys.txt", "--metrics-open"])
    assert args.api_key == ["fixture-first", "fixture-second"]
    assert args.api_key_file == "keys.txt" and args.metrics_open is True


