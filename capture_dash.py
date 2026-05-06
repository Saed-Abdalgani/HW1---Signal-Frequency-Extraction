import asyncio
import os
import random
from contextlib import suppress

from playwright.async_api import async_playwright


async def main():
    os.makedirs('screenshots', exist_ok=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 900})
        await page.goto('http://localhost:8050')
        await page.wait_for_selector('.dashboard-container', timeout=10000)

        # Give it a second to render initial graphs
        await asyncio.sleep(2)

        tabs = ['SIGNALS', 'T-SNE 3D', 'PCA 3D', 'FFT SPECTRUM']

        for count in range(1, 25):
            print(f"Generating unique state for screenshot {count}...")

            # Switch to a random tab to show variety
            tab = random.choice(tabs)
            with suppress(Exception):
                await page.click(f'text={tab}', timeout=2000)

            # Randomly adjust global parameters
            with suppress(Exception):
                await page.click('#fs-slider .rc-slider', position={'x': random.randint(20, 150), 'y': 5}, timeout=1000)
                await page.click('#n-cycles .rc-slider', position={'x': random.randint(20, 150), 'y': 5}, timeout=1000)
                await page.click('#bw-slider .rc-slider', position={'x': random.randint(20, 150), 'y': 5}, timeout=1000)

            # Randomly adjust parameters for all 4 sinusoids
            for sin_idx in range(1, 5):
                with suppress(Exception):
                    await page.click(f'#freq-{sin_idx} .rc-slider', position={'x': random.randint(10, 180), 'y': 5}, timeout=1000)
                    await page.click(f'#amp-{sin_idx} .rc-slider', position={'x': random.randint(10, 180), 'y': 5}, timeout=1000)
                    await page.click(f'#phase-{sin_idx} .rc-slider', position={'x': random.randint(10, 180), 'y': 5}, timeout=1000)

                    if random.random() > 0.4:
                        await page.click(f'#sigma-{sin_idx} .rc-slider', position={'x': random.randint(10, 100), 'y': 5}, timeout=1000)

            # Wait for Plotly to render the new state
            await asyncio.sleep(1.5)
            await page.screenshot(path=f'screenshots/dash_{count}.png')
            print(f"Saved dash_{count}.png")

        await browser.close()
        print("Done capturing completely unique screenshots.")

asyncio.run(main())
