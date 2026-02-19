#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from gtts import gTTS
import subprocess
import os

class TTSNode(Node):
    def __init__(self):
        super().__init__('tts_node')
        self.subscription = self.create_subscription(
            String,
            '/tts_text',  # Topic to subscribe
            self.tts_callback,
            10
        )
        self.get_logger().info("TTS Node started, waiting for messages...")

    def tts_callback(self, msg: String):
        text = msg.data.strip()
        if not text:
            return

        self.get_logger().info(f"Speaking: {text}")

        # Generate MP3 via gTTS
        tts = gTTS(text=text, lang='en-uk')
        tmp_mp3 = "/tmp/tts.mp3"
        tmp_wav = "/tmp/tts.wav"
        tts.save(tmp_mp3)

        # Convert MP3 -> WAV for ALSA
        subprocess.run(['ffmpeg', '-y', '-i', tmp_mp3, tmp_wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Play via ALSA (plughw converts mono->stereo and sample rate)
        subprocess.run(['aplay', '-D', 'plughw:1,0', tmp_wav], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Clean up temporary files
        try:
            os.remove(tmp_mp3)
            os.remove(tmp_wav)
        except FileNotFoundError:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()