from app.camera.video_reader import VideoReader


class RTSPReader(VideoReader):
    """Named RTSP reader for future RTSP-specific reconnect logic."""

    def __init__(self, source: str) -> None:
        if not source.lower().startswith("rtsp://"):
            raise ValueError("RTSPReader requires an rtsp:// source")
        super().__init__(source)
