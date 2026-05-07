"""
Recapture dash_22/23 using correct Radix portal discovery via JS.
"""
import asyncio

# pyrefly: ignore [missing-import]
from playwright.async_api import async_playwright


async def select_radix_dropdown(page, dropdown_id, option_text):
    """Click Radix trigger then find+click option in the portal via JS."""
    # Click the trigger
    trigger = page.locator(f"#{dropdown_id}")
    await trigger.click()
    await page.wait_for_timeout(1000)

    # Use JS to find the menu via aria-controls and click the right option
    clicked = await page.evaluate(f"""(text) => {{
        // Get the controls id from the trigger
        const trigger = document.getElementById('{dropdown_id}');
        if (!trigger) return 'no trigger';
        const menuId = trigger.getAttribute('aria-controls');
        if (!menuId) return 'no aria-controls';

        // Find the menu element by id (using JS, not CSS)
        const menu = document.getElementById(menuId);
        if (!menu) return 'no menu: ' + menuId;

        // Find child with matching text
        const items = menu.querySelectorAll('[role="option"], .dash-dropdown-option, label');
        for (const item of items) {{
            if (item.textContent.trim() === text) {{
                item.dispatchEvent(new PointerEvent('pointerdown', {{bubbles: true, cancelable: true}}));
                item.dispatchEvent(new MouseEvent('mousedown', {{bubbles: true, cancelable: true}}));
                item.click();
                return 'clicked: ' + item.textContent.trim();
            }}
        }}
        return 'option not found. items: ' + Array.from(items).map(i=>i.textContent.trim()).join('|');
    }}""", option_text)
    print(f"  JS click result: {clicked}")
    await page.wait_for_timeout(2500)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1536, "height": 900})

        print("Loading dashboard...")
        await page.goto("http://localhost:8050", wait_until="networkidle", timeout=30000)
        await page.wait_for_selector(".js-plotly-plot", timeout=15000)
        await page.wait_for_timeout(3000)

        # ─────────────────────────────────────────────────────
        # dash_22: Highpass filter
        # ─────────────────────────────────────────────────────
        print("Screenshot dash_22: Highpass filter...")
        await select_radix_dropdown(page, "filter-dropdown", "Highpass")
        current = await page.locator("#filter-dropdown").text_content()
        print(f"  Dropdown text: {current!r}")
        await page.screenshot(path="screenshots/dash_22.png")
        print("  Saved dash_22.png")

        # ─────────────────────────────────────────────────────
        # dash_23: Lowpass filter
        # ─────────────────────────────────────────────────────
        print("Screenshot dash_23: Lowpass filter...")
        await select_radix_dropdown(page, "filter-dropdown", "Lowpass")
        current2 = await page.locator("#filter-dropdown").text_content()
        print(f"  Dropdown text: {current2!r}")
        await page.screenshot(path="screenshots/dash_23.png")
        print("  Saved dash_23.png")

        await browser.close()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
