
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Slack API Token (OAuth Token을 입력)
SLACK_TOKEN = ""

# Slack 클라이언트 생성
client = WebClient(token=SLACK_TOKEN)

def send_slack_message(channel, text):
    """Slack 특정 채널에 메시지를 전송하는 함수"""
    try:
        response = client.chat_postMessage(channel=channel, text=text)
        print("Slack 메시지 전송 성공!")
    except SlackApiError as e:
        print(f"Slack 메시지 전송 실패: {e.response['error']}")
        

send_slack_message("experiment-notification", "FollowTable (Department) 평가 완료!")