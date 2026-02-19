import json

import ollama
import rclpy
import requests
from rclpy.node import Node
from std_msgs.msg import String


class LLMNode(Node):
    def __init__(self):
        super().__init__("llm_node")
        self.publisher_ = self.create_publisher(String, "/tts_text", 10)
        self.get_logger().info("LLM Node started.")

    def run(self):
        while True:
            u_input = input("\nYou: ")
            if u_input.lower() == "exit":
                break

            response = ollama.chat(
                model="nlp_llama",
                messages=[
                    {"role": "user", "content": u_input},
                ],
            )

            reply = response["message"]["content"]

            parsed = json.loads(reply)
            answer = parsed["response"]

            print(answer)
            print(f"\nLLM: {reply}")

            msg = String()
            msg.data = answer
            self.publisher_.publish(msg)


def main():
    rclpy.init()
    node = LLMNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
