"""
Strava OAuth2 One-Click Token Generator
Automatically launches browser for authorization and captures refresh_token.
"""

import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import requests
from config import BASE_DIR

auth_code = None

class OAuthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global auth_code
        query = parse_qs(urlparse(self.path).query)
        if "code" in query:
            auth_code = query["code"][0]
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write("""
                <html>
                <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
                    <h1 style="color: #22c55e;">🎉 인증 성공!</h1>
                    <p>스트라바 계정 연동 권한이 승인되었습니다. 이 창을 닫고 터미널로 돌아가세요.</p>
                </body>
                </html>
            """.encode("utf-8"))
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Failed to get authorization code.")

    def log_message(self, format, *args):
        pass  # Suppress server logs


def get_strava_tokens(client_id: str, client_secret: str):
    port = 8000
    redirect_uri = f"http://localhost:{port}/exchange_token"
    scope = "read,activity:read_all"

    auth_url = (
        f"https://www.strava.com/oauth/authorize?"
        f"client_id={client_id}&"
        f"response_type=code&"
        f"redirect_uri={redirect_uri}&"
        f"approval_prompt=force&"
        f"scope={scope}"
    )

    print("\n" + "="*60)
    print("🚴 스트라바(Strava) 권한 승인을 위해 브라우저를 엽니다...")
    print(f"브라우저가 열리지 않으면 아래 링크를 직접 열어주세요:\n{auth_url}")
    print("="*60 + "\n")

    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", port), OAuthHandler)
    server.handle_request()

    if not auth_code:
        print("❌ 인증 코드를 가져오지 못했습니다.")
        return

    print("🔑 인증 코드를 토큰으로 교환하는 중...")
    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": auth_code,
            "grant_type": "authorization_code",
        },
    )
    if resp.status_code != 200:
        print(f"❌ 토큰 교환 실패: {resp.text}")
        return

    data = resp.json()
    refresh_token = data.get("refresh_token")
    athlete = data.get("athlete", {})
    athlete_name = f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip()

    print("\n" + "🎉"*20)
    print(f"✅ 연동 성공! 선수 이름: {athlete_name}")
    print(f"📌 발급된 Refresh Token: {refresh_token}")
    print("🎉"*20 + "\n")

    # Update .env file automatically
    env_path = BASE_DIR / ".env"
    env_content = ""
    if env_path.exists():
        env_content = env_path.read_text(encoding="utf-8")

    lines = env_content.splitlines()
    updated = {}
    new_lines = []
    keys_to_update = {
        "STRAVA_CLIENT_ID": client_id,
        "STRAVA_CLIENT_SECRET": client_secret,
        "STRAVA_REFRESH_TOKEN": refresh_token,
    }

    for line in lines:
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            k = k.strip()
            if k in keys_to_update:
                new_lines.append(f"{k}={keys_to_update[k]}")
                updated[k] = True
                continue
        new_lines.append(line)

    for k, v in keys_to_update.items():
        if k not in updated:
            new_lines.append(f"{k}={v}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print("💾 .env 파일에 Client ID, Client Secret, Refresh Token이 자동으로 저장되었습니다!")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        cid = sys.argv[1]
        csec = sys.argv[2]
    else:
        cid = input("스트라바 Client ID를 입력하세요: ").strip()
        csec = input("스트라바 Client Secret을 입력하세요: ").strip()

    get_strava_tokens(cid, csec)
