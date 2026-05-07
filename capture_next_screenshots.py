import asyncio
from playwright.async_api import async_playwright

async def capture_browser_shots():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1536, "height": 900})

        print("Loading dashboard...")
        await page.goto("http://localhost:8050", wait_until="networkidle", timeout=30000)
        await page.wait_for_selector(".js-plotly-plot", timeout=15000)
        await page.wait_for_timeout(3000)

        # PRIORITY 5: Header Bar Close-up (FR-9.3)
        # On the SIGNALS tab with multiple sinusoids active
        print("Enabling Sin 2 and Sin 3...")
        await page.locator("#mix-2").click()
        await page.locator("#mix-3").click()
        await page.wait_for_timeout(2000)
        
        print("Capturing dash_27.png (Header metrics)...")
        await page.locator(".app-header").screenshot(path="screenshots/dash_27.png")
        
        # PRIORITY 1: FFT with BPF Bandwidth Shading (FR-12)
        print("Enabling BPF for Sin 1...")
        await page.locator("#bpf-1").click()
        
        print("Adjusting BW slider to ~15 Hz...")
        # Focus the slider and use right arrow to increase from 10 to 15 (50 steps of 0.1)
        # Actually it's easier to evaluate JS
        await page.evaluate("""() => {
            const slider = document.getElementById('bw-slider');
            // Since it's a Dash component we might not be able to just set value.
            // Let's try to just click on the track to the right.
        }""")
        
        # Another way: Bounding box of slider track
        slider = await page.locator("#bw-slider").bounding_box()
        # Click a bit to the right of the center
        if slider:
            await page.mouse.click(slider['x'] + slider['width'] * 0.2, slider['y'] + slider['height'] / 2)
            
        await page.wait_for_timeout(1000)
        
        print("Switching to FFT SPECTRUM tab...")
        await page.locator("text=FFT SPECTRUM").click()
        await page.wait_for_timeout(3000)
        
        print("Capturing dash_26.png (FFT BPF Shaded)...")
        await page.screenshot(path="screenshots/dash_26.png")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(capture_browser_shots())
