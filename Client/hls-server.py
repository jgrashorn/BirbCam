import ffmpeg
import os
from flask import Flask, send_from_directory
import threading

def reencode_udp_to_hls(input_url, output_path, segment_time=4, playlist_size=5):
    """
    Re-encodes a UDP stream to HLS format.

    Args:
    - input_url (str): The URL of the UDP stream.
    - output_path (str): The directory to save the HLS files.
    - segment_time (int): Duration of each HLS segment in seconds.
    - playlist_size (int): Number of segments to keep in the HLS playlist.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Build the FFmpeg HLS re-encoding command
    (
        ffmpeg
        .input(input_url)  # The UDP stream input
        .output(
            os.path.join(output_path, 'stream.m3u8'),
            format='hls',
            hls_time=segment_time,
            hls_list_size=playlist_size,
            hls_flags='delete_segments',
            vcodec='libx264',   # Video codec
            acodec='aac'        # Audio codec
        )
        .run()
    )

# Usage example
input_url = 'udp://@192.168.178.83:8081?pkt_size=1316'  # Replace with your actual UDP stream URL
output_path = './hls_output/'  # Replace with the path where you want to save HLS files

app = Flask(__name__)

@app.route('/hls/<path:filename>')
def serve_hls(filename):
    with open("log.log") as f:
        f.writelines("streaming...\n")
    """Serves the HLS .m3u8 and .ts files"""
    return send_from_directory(output_path, filename)

def start_flask_server():
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)

if __name__ == "__main__":
    # Start the re-encoding process in a separate thread
    threading.Thread(target=reencode_udp_to_hls, args=(input_url, output_path), daemon=True).start()

    # Start the Flask server
    start_flask_server()