import { expect, test } from '@playwright/test';

type MockWidgetResponse = {
  delayMs?: number;
  endpoint: string;
  title: string;
  html: string;
};

const widgetKey = 'test-widget-key';
const createPrompt = "Show me today's crude oil price versus gold";
const modifyPrompt = 'Turn this into a flight delay summary instead';

function buildTallWidgetHtml(title: string, markerClass: string): string {
  const rows = Array.from({ length: 24 }, (_, index) => {
    const value = 80 + index;
    return `
      <div class="market-item" style="padding: 12px; border: 1px solid rgba(255,255,255,0.08); border-radius: 10px;">
        <div class="market-item-name">${title} ${index + 1}</div>
        <div class="market-item-price">$${value}</div>
      </div>
    `;
  }).join('');

  return `
    <div class="${markerClass}" data-widget-marker="${markerClass}" style="display:grid;gap:12px;">
      <div
        data-escape-banner="true"
        style="position:fixed;top:0;left:0;width:200vw;height:44px;background:#ff4444;color:#fff;z-index:9999;"
      >
        escape banner
      </div>
      <div class="economic-header" style="display:grid;gap:4px;">
        <strong>${title}</strong>
        <span>Live WorldMonitor snapshot</span>
      </div>
      <div class="economic-grid" style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;">
        ${rows}
      </div>
      <div class="economic-footer">
        <span>Source: WorldMonitor</span>
      </div>
    </div>
  `;
}

function buildWidgetSseResponse({ endpoint, title, html }: MockWidgetResponse): string {
  return [
    { type: 'tool_call', endpoint },
    { type: 'html_complete', html },
    { type: 'done', title },
  ]
    .map((payload) => `data: ${JSON.stringify(payload)}\n\n`)
    .join('');
}

async function installWidgetAgentMocks(
  page: Parameters<typeof test>[0]['page'],
  responses: MockWidgetResponse[],
  requestBodies: unknown[] = [],
  healthDelayMs = 0,
): Promise<void> {
  await page.route('**/widget-agent/health', async (route) => {
    if (healthDelayMs > 0) {
      await new Promise((resolve) => setTimeout(resolve, healthDelayMs));
    }

    expect(route.request().headers()['x-widget-key']).toBe(widgetKey);
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        ok: true,
        agentEnabled: true,
        widgetKeyConfigured: true,
        anthropicConfigured: true,
      }),
    });
  });

  let responseIndex = 0;
  await page.route('**/widget-agent', async (route) => {
    const body = route.request().postDataJSON();
    requestBodies.push(body);

    const response = responses[responseIndex];
    if (!response) {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Unexpected extra widget-agent call' }),
      });
      return;
    }

    responseIndex += 1;
    if ((response.delayMs ?? 0) > 0) {
      await new Promise((resolve) => setTimeout(resolve, response.delayMs));
    }

    await route.fulfill({
      status: 200,
      contentType: 'text/event-stream',
      headers: {
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      },
      body: buildWidgetSseResponse(response),
    });
  });
}

test.describe('AI widget builder', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript((key) => {
      if (!sessionStorage.getItem('__widget_e2e_init__')) {
        localStorage.clear();
        sessionStorage.clear();
        localStorage.setItem('worldmonitor-variant', 'happy');
        localStorage.setItem('wm-widget-key', key);
        sessionStorage.setItem('__widget_e2e_init__', '1');
        return;
      }

      if (!localStorage.getItem('wm-widget-key')) {
        localStorage.setItem('wm-widget-key', key);
      }
    }, widgetKey);
  });

  test('creates a widget through the live modal flow and persists it after reload', async ({ page }) => {
    const createHtml = buildTallWidgetHtml('Oil vs Gold', 'oil-gold-widget');
    await installWidgetAgentMocks(
      page,
      [
        {
          delayMs: 250,
          endpoint: '/rpc/worldmonitor.markets.v1.MarketsService/GetCommodities',
          title: 'Oil vs Gold',
          html: createHtml,
        },
      ],
      [],
      500,
    );

    await page.goto('/');
    await expect(page.locator('#panelsGrid .ai-widget-block')).toBeVisible({ timeout: 30000 });

    await page.locator('#panelsGrid .ai-widget-block').click();

    const modal = page.locator('.widget-chat-modal');
    const sendButton = modal.locator('.widget-chat-send');
    const input = modal.locator('.widget-chat-input');
    const preview = modal.locator('.widget-chat-preview');
    const footer = modal.locator('.widget-chat-footer');
    const footerAction = footer.locator('.widget-chat-action-btn');

    await expect(modal).toBeVisible();
    await expect(modal.locator('.widget-chat-layout')).toBeVisible();
    await expect(modal.locator('.widget-chat-sidebar')).toBeVisible();
    await expect(modal.locator('.widget-chat-main')).toBeVisible();

    await expect(modal.locator('.widget-chat-example-chip')).toHaveCount(4);
    await modal.locator('.widget-chat-example-chip').first().click();
    await expect(input).toHaveValue(createPrompt);

    await expect(modal.locator('.widget-chat-readiness')).toContainText('Connected to the widget agent');
    await expect(preview).toContainText('Describe the widget you want');
    await expect(sendButton).toBeEnabled();

    const sidebarBox = await modal.locator('.widget-chat-sidebar').boundingBox();
    const mainBox = await modal.locator('.widget-chat-main').boundingBox();
    expect(sidebarBox?.width ?? 0).toBeGreaterThan(280);
    expect(mainBox?.width ?? 0).toBeGreaterThan(320);

    await sendButton.click();

    await expect(preview.locator('.widget-chat-preview-frame')).toBeVisible({ timeout: 30000 });
    await expect(preview).toContainText('Oil vs Gold');
    await expect(preview.locator('.wm-widget-shell')).toBeVisible();
    await expect(preview.locator('.wm-widget-generated')).toBeVisible();
    await expect(footerAction).toBeEnabled();

    const footerBefore = await footer.boundingBox();
    await preview.evaluate((element) => {
      element.scrollTop = element.scrollHeight;
    });
    const footerAfter = await footer.boundingBox();
    expect(Math.abs((footerAfter?.y ?? 0) - (footerBefore?.y ?? 0))).toBeLessThan(2);
    await expect(footerAction).toBeVisible();

    await footerAction.click();

    const widgetPanel = page.locator('.custom-widget-panel', {
      has: page.locator('.panel-title', { hasText: 'Oil vs Gold' }),
    });
    await expect(widgetPanel).toBeVisible({ timeout: 20000 });
    await expect(widgetPanel.locator('.wm-widget-shell')).toBeVisible();
    await expect(widgetPanel.locator('.wm-widget-generated')).toBeVisible();

    const containment = await widgetPanel.locator('.wm-widget-generated').evaluate((element) => {
      const style = getComputedStyle(element);
      return {
        contain: style.contain,
        overflowX: style.overflowX,
        overflowY: style.overflowY,
      };
    });
    expect(containment.contain).toContain('layout');
    expect(containment.contain).toContain('paint');
    expect(['clip', 'hidden']).toContain(containment.overflowX);
    expect(['clip', 'hidden']).toContain(containment.overflowY);

    const bannerPosition = await widgetPanel.evaluate((panel) => {
      const panelRect = panel.getBoundingClientRect();
      const banner = panel.querySelector('[data-escape-banner="true"]') as HTMLElement | null;
      const bannerRect = banner?.getBoundingClientRect() ?? null;
      return { panelRect, bannerRect };
    });
    expect(bannerPosition.bannerRect).not.toBeNull();
    expect(bannerPosition.bannerRect!.top).toBeGreaterThanOrEqual(bannerPosition.panelRect.top - 1);
    expect(bannerPosition.bannerRect!.left).toBeGreaterThanOrEqual(bannerPosition.panelRect.left - 1);

    await page.reload();
    await expect(page.locator('.custom-widget-panel', {
      has: page.locator('.panel-title', { hasText: 'Oil vs Gold' }),
    })).toBeVisible({ timeout: 20000 });

    const storedWidgets = await page.evaluate(() => {
      return JSON.parse(localStorage.getItem('wm-custom-widgets') || '[]') as Array<{ title: string }>;
    });
    expect(storedWidgets.some((entry) => entry.title === 'Oil vs Gold')).toBe(true);
  });

  test('supports modify, keeps session history, exposes touch-sized controls, and cleans storage on delete', async ({ page }) => {
    const requestBodies: unknown[] = [];
    await installWidgetAgentMocks(page, [
      {
        endpoint: '/rpc/worldmonitor.markets.v1.MarketsService/GetCommodities',
        title: 'Oil vs Gold',
        html: buildTallWidgetHtml('Oil vs Gold', 'oil-gold-widget'),
      },
      {
        endpoint: '/rpc/worldmonitor.aviation.v1.AviationService/GetAirportDelays',
        title: 'Flight Delay Watch',
        html: buildTallWidgetHtml('Flight Delay Watch', 'flight-delay-widget'),
      },
    ], requestBodies);

    await page.goto('/');
    await expect(page.locator('#panelsGrid .ai-widget-block')).toBeVisible({ timeout: 30000 });

    await page.locator('#panelsGrid .ai-widget-block').click();
    const modal = page.locator('.widget-chat-modal');
    await expect(modal.locator('.widget-chat-readiness')).toContainText('Connected to the widget agent');

    await modal.locator('.widget-chat-input').fill(createPrompt);
    await modal.locator('.widget-chat-send').click();
    await expect(modal.locator('.widget-chat-action-btn')).toBeEnabled({ timeout: 30000 });
    await modal.locator('.widget-chat-action-btn').click();

    const widgetPanel = page.locator('.custom-widget-panel', {
      has: page.locator('.panel-title', { hasText: 'Oil vs Gold' }),
    });
    await expect(widgetPanel).toBeVisible({ timeout: 20000 });

    const modifyButton = widgetPanel.locator('.panel-widget-chat-btn');
    const colorButton = widgetPanel.locator('.widget-color-btn');
    await expect(modifyButton).toBeVisible();
    await expect(colorButton).toBeVisible();

    const controlSizes = await widgetPanel.evaluate((panel) => {
      const modify = panel.querySelector('.panel-widget-chat-btn') as HTMLElement | null;
      const color = panel.querySelector('.widget-color-btn') as HTMLElement | null;
      const modifyRect = modify?.getBoundingClientRect();
      const colorRect = color?.getBoundingClientRect();
      return {
        modifyWidth: modifyRect?.width ?? 0,
        modifyHeight: modifyRect?.height ?? 0,
        colorWidth: colorRect?.width ?? 0,
        colorHeight: colorRect?.height ?? 0,
      };
    });
    expect(controlSizes.modifyWidth).toBeGreaterThanOrEqual(32);
    expect(controlSizes.modifyHeight).toBeGreaterThanOrEqual(32);
    expect(controlSizes.colorWidth).toBeGreaterThanOrEqual(32);
    expect(controlSizes.colorHeight).toBeGreaterThanOrEqual(32);

    const initialAccent = await colorButton.evaluate((button) => getComputedStyle(button).backgroundColor);
    await colorButton.click();
    const updatedAccent = await colorButton.evaluate((button) => getComputedStyle(button).backgroundColor);
    expect(updatedAccent).not.toBe(initialAccent);

    await modifyButton.click();
    const modifyModal = page.locator('.widget-chat-modal');
    await expect(modifyModal).toBeVisible();
    await expect(modifyModal.locator('.widget-chat-messages')).toContainText(createPrompt);
    await expect(modifyModal.locator('.widget-chat-messages')).toContainText('Generated widget: Oil vs Gold');
    await expect(modifyModal.locator('.widget-chat-preview')).toContainText('Oil vs Gold');

    await modifyModal.locator('.widget-chat-input').fill(modifyPrompt);
    await modifyModal.locator('.widget-chat-send').click();
    await expect(modifyModal.locator('.widget-chat-action-btn')).toBeEnabled({ timeout: 30000 });
    await expect(modifyModal.locator('.widget-chat-preview')).toContainText('Flight Delay Watch');
    await modifyModal.locator('.widget-chat-action-btn').click();

    const updatedPanel = page.locator('.custom-widget-panel', {
      has: page.locator('.panel-title', { hasText: 'Flight Delay Watch' }),
    });
    await expect(updatedPanel).toBeVisible({ timeout: 20000 });

    const storedWidgetMeta = await page.evaluate(() => {
      const widgets = JSON.parse(localStorage.getItem('wm-custom-widgets') || '[]') as Array<{
        id: string;
        title: string;
      }>;
      return widgets.find((entry) => entry.title === 'Flight Delay Watch') ?? null;
    });
    expect(storedWidgetMeta).not.toBeNull();

    const secondRequest = requestBodies[1] as {
      conversationHistory?: Array<{ role: string; content: string }>;
      currentHtml?: string | null;
    } | undefined;
    expect(secondRequest?.currentHtml).toContain('oil-gold-widget');
    expect(secondRequest?.conversationHistory?.some((entry) => entry.content.includes(createPrompt))).toBe(true);
    expect(secondRequest?.conversationHistory?.some((entry) => entry.content.includes('Generated widget: Oil vs Gold'))).toBe(true);

    await page.evaluate((widgetId: string) => {
      localStorage.setItem('worldmonitor-panel-spans', JSON.stringify({ [widgetId]: 2 }));
      localStorage.setItem('worldmonitor-panel-col-spans', JSON.stringify({ [widgetId]: 3 }));
    }, storedWidgetMeta!.id);

    await page.evaluate(() => {
      window.confirm = () => true;
    });
    await updatedPanel.locator('.panel-close-btn').evaluate((button: HTMLButtonElement) => {
      button.click();
    });
    await expect(updatedPanel).toHaveCount(0);

    const cleanedStorage = await page.evaluate(() => {
      return {
        widgets: localStorage.getItem('wm-custom-widgets'),
        rowSpans: localStorage.getItem('worldmonitor-panel-spans'),
        colSpans: localStorage.getItem('worldmonitor-panel-col-spans'),
      };
    });
    expect(cleanedStorage.widgets).toBe('[]');
    expect(cleanedStorage.rowSpans).toBeNull();
    expect(cleanedStorage.colSpans).toBeNull();

    await page.reload();
    await expect(page.locator('.custom-widget-panel')).toHaveCount(0);
  });
});
