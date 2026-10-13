"""Utility script to capture genuine screenshots of the InferPath Streamlit dashboard via CDP."""

import asyncio
import base64
import json
import subprocess
from pathlib import Path

import httpx
import websockets

EDGE_EXE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9222
STREAMLIT_URL = "http://127.0.0.1:8501"
OUTPUT_DIR = Path("docs/img")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


async def send_cdp(ws, method: str, params: dict | None = None, req_id: int = 1) -> dict:
    payload = {"id": req_id, "method": method, "params": params or {}}
    await ws.send(json.dumps(payload))
    while True:
        resp = await ws.recv()
        data = json.loads(resp)
        if data.get("id") == req_id:
            return data


async def capture_tab(ws, filename: str, req_id: int):
    print(f"Capturing screenshot: {filename}...")
    res = await send_cdp(ws, "Page.captureScreenshot", {"format": "png"}, req_id=req_id)
    img_b64 = res["result"]["data"]
    img_bytes = base64.b64decode(img_b64)
    out_path = OUTPUT_DIR / filename
    out_path.write_bytes(img_bytes)
    print(f"Saved {out_path} ({len(img_bytes)} bytes)")


async def main():
    # 1. Launch headless Edge with remote debugging port
    temp_dir = Path("temp_cdp_profile")
    temp_dir.mkdir(exist_ok=True)

    args = [
        EDGE_EXE,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={temp_dir.resolve()}",
        "--window-size=1280,960",
        STREAMLIT_URL,
    ]
    proc = subprocess.Popen(args)
    print(f"Started Edge PID={proc.pid} on port {PORT}")

    try:
        # 2. Wait for CDP endpoint to respond
        ws_url = None
        for _ in range(20):
            await asyncio.sleep(0.5)
            try:
                r = httpx.get(f"http://127.0.0.1:{PORT}/json/list", timeout=1.0)
                if r.status_code == 200 and r.json():
                    pages = [p for p in r.json() if p.get("type") == "page"]
                    if pages:
                        ws_url = pages[0]["webSocketDebuggerUrl"]
                        break
            except Exception:
                pass

        if not ws_url:
            print("Failed to find webSocketDebuggerUrl")
            return

        print(f"Connecting to CDP: {ws_url}")
        async with websockets.connect(ws_url, max_size=10_000_000) as ws:
            # Wait for Streamlit WebSocket to connect and render DOM
            print("Waiting for Streamlit to render initial view (5 seconds)...")
            await asyncio.sleep(5.0)

            async def click_tab(keyword: str, req_id_prefix: int):
                js = f"""
                (() => {{
                    const candidates = Array.from(document.querySelectorAll('button, [data-baseweb="tab"], [role="tab"]'));
                    const target = candidates.find(t => (t.innerText || t.textContent || '').includes('{keyword}'));
                    if (target) {{
                        target.click();
                        target.dispatchEvent(new MouseEvent('click', {{ bubbles: true, cancelable: true }}));
                        return target.innerText;
                    }}
                    return null;
                }})()
                """
                res = await send_cdp(
                    ws,
                    "Runtime.evaluate",
                    {"expression": js, "returnByValue": True},
                    req_id=req_id_prefix,
                )
                print(f"Clicked tab '{keyword}': result={res.get('result', {}).get('value')}")
                await asyncio.sleep(3.0)

            # Tab 1: Overview
            await capture_tab(ws, "dashboard_overview.png", req_id=101)

            # Tab 2: Routing
            await click_tab("Routing", 102)
            await capture_tab(ws, "dashboard_routing.png", req_id=103)

            # Tab 3: Exact Cache
            await click_tab("Cache", 104)
            await capture_tab(ws, "dashboard_cache.png", req_id=105)

            # Tab 4: Resources
            await click_tab("Resources", 106)
            await capture_tab(ws, "dashboard_resources.png", req_id=107)

            # Tab 5: Failures
            await click_tab("Failure", 108)
            await capture_tab(ws, "dashboard_failures.png", req_id=109)

            # Tab 6: Live Demo
            await click_tab("Live Demo", 110)
            await capture_tab(ws, "dashboard_live_demo.png", req_id=111)

            # Trigger Route & Execute Request in Live Demo
            click_submit_js = """
            (() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const submitBtn = btns.find(b => b.innerText && b.innerText.includes('Route & Execute Request'));
                if (submitBtn) {
                    submitBtn.click();
                    submitBtn.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
                    return true;
                }
                return false;
            })()
            """
            await send_cdp(ws, "Runtime.evaluate", {"expression": click_submit_js}, req_id=112)
            print("Submitted Live Demo request. Waiting for response (3 seconds)...")
            await asyncio.sleep(3.0)
            scroll_js = """
            (() => {
                const main = document.querySelector('section.main') || window;
                main.scrollTo(0, 1000);
            })()
            """
            await send_cdp(ws, "Runtime.evaluate", {"expression": scroll_js}, req_id=113)
            await asyncio.sleep(1.5)
            await capture_tab(ws, "dashboard_live_demo_result.png", req_id=114)

    finally:
        print("Terminating Edge...")
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
