from vocnu_engine import render


def test_reencoding_is_needed_only_when_the_container_cannot_hold_the_picture():
    assert not render.needs_reencode('mp4', 'h264') and not render.needs_reencode('mov', 'hevc')
    assert render.needs_reencode('mp4', 'vp9') and render.needs_reencode('webm', 'h264')
    assert not render.needs_reencode('webm', 'vp9') and not render.needs_reencode('avi', 'h264')
    assert render.needs_reencode('avi', 'hevc')
    assert not render.needs_reencode('mkv', 'vp9') and not render.needs_reencode('mkv', None)
    assert render.needs_reencode('mkv', 'h264', burned=True)
