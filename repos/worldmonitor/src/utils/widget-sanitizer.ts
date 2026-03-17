import DOMPurify from 'dompurify';

const PURIFY_CONFIG = {
  ALLOWED_TAGS: [
    'div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
    'ul', 'ol', 'li', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
    'strong', 'em', 'b', 'i', 'br', 'hr', 'small',
    'svg', 'path', 'circle', 'rect', 'line', 'polyline', 'polygon', 'text', 'tspan',
  ],
  ALLOWED_ATTR: [
    'class', 'style', 'title', 'aria-label',
    'viewBox', 'fill', 'stroke', 'stroke-width',
    'd', 'cx', 'cy', 'r', 'x', 'y', 'width', 'height', 'points',
    'xmlns',
  ],
  FORBID_TAGS: ['button', 'input', 'form', 'select', 'textarea', 'script', 'iframe', 'object', 'embed'],
  ALLOW_DATA_ATTR: false,
  FORCE_BODY: true,
};

const UNSAFE_STYLE_PATTERN = /url\s*\(|expression\s*\(|javascript\s*:|@import|behavior\s*:/i;

DOMPurify.addHook('uponSanitizeAttribute', (_node, data) => {
  if (data.attrName === 'style' && UNSAFE_STYLE_PATTERN.test(data.attrValue)) {
    data.keepAttr = false;
  }
});

export function sanitizeWidgetHtml(html: string): string {
  return DOMPurify.sanitize(html, PURIFY_CONFIG) as unknown as string;
}

export function wrapWidgetHtml(html: string, extraClass = ''): string {
  const shellClass = ['wm-widget-shell', extraClass].filter(Boolean).join(' ');
  return `
    <div class="${shellClass}">
      <div class="wm-widget-body">
        <div class="wm-widget-generated">${sanitizeWidgetHtml(html)}</div>
      </div>
    </div>
  `;
}
