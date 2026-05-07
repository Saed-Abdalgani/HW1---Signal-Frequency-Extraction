import asyncio
from playwright.async_api import async_playwright

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            background-color: #0d1117;
            color: #c9d1d9;
            font-family: 'Consolas', 'Courier New', monospace;
            padding: 20px;
            margin: 0;
            line-height: 1.5;
        }}
        pre {{
            white-space: pre-wrap;
            margin: 0;
        }}
        .green {{ color: #7ee787; }}
        .red {{ color: #ff7b72; }}
        .yellow {{ color: #d2a8ff; }}
    </style>
</head>
<body>
<pre>{content}</pre>
</body>
</html>
"""

async def capture_terminal(file_path, output_png):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Create HTML
    html_content = HTML_TEMPLATE.format(content=text.replace('<', '&lt;').replace('>', '&gt;'))
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1000, "height": 800})
        await page.set_content(html_content)
        await page.wait_for_timeout(500)
        await page.locator('body').screenshot(path=output_png)
        await browser.close()
    print(f"Captured {output_png}")

async def main():
    await capture_terminal('out_pytest.txt', 'screenshots/pytest_coverage.png')
    await capture_terminal('out_ruff.txt', 'screenshots/ruff_clean.png')
    
    # Extract only the relevant part of out_all.txt for training output
    with open('out_all.txt', 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        
    start_idx = 0
    for i, line in enumerate(lines):
        if 'freq_extractor.services.ml.model_mlp' in line and 'Epoch 1/50' in line:
            start_idx = max(0, i - 2)
            break
            
    train_text = "".join(lines[start_idx:start_idx+100])
    # save to train.txt
    with open('train.txt', 'w', encoding='utf-8') as f:
        f.write(train_text)
        
    await capture_terminal('train.txt', 'screenshots/terminal_training.png')

if __name__ == '__main__':
    asyncio.run(main())
