import { loadFromStorage, saveToStorage } from '@/utils';
import { sanitizeWidgetHtml } from '@/utils/widget-sanitizer';

const STORAGE_KEY = 'wm-custom-widgets';
const PANEL_SPANS_KEY = 'worldmonitor-panel-spans';
const PANEL_COL_SPANS_KEY = 'worldmonitor-panel-col-spans';
const MAX_WIDGETS = 10;
const MAX_HISTORY = 10;
const MAX_HTML_CHARS = 50_000;

export interface CustomWidgetSpec {
  id: string;
  title: string;
  html: string;
  prompt: string;
  accentColor: string | null;
  conversationHistory: Array<{ role: 'user' | 'assistant'; content: string }>;
  createdAt: number;
  updatedAt: number;
}

export function loadWidgets(): CustomWidgetSpec[] {
  return loadFromStorage<CustomWidgetSpec[]>(STORAGE_KEY, []);
}

export function saveWidget(spec: CustomWidgetSpec): void {
  const trimmed: CustomWidgetSpec = {
    ...spec,
    html: sanitizeWidgetHtml(spec.html.slice(0, MAX_HTML_CHARS)),
    conversationHistory: spec.conversationHistory.slice(-MAX_HISTORY),
  };
  const existing = loadWidgets().filter(w => w.id !== trimmed.id);
  const updated = [...existing, trimmed].slice(-MAX_WIDGETS);
  saveToStorage(STORAGE_KEY, updated);
}

export function deleteWidget(id: string): void {
  const updated = loadWidgets().filter(w => w.id !== id);
  saveToStorage(STORAGE_KEY, updated);
  cleanSpanEntry(PANEL_SPANS_KEY, id);
  cleanSpanEntry(PANEL_COL_SPANS_KEY, id);
}

export function getWidget(id: string): CustomWidgetSpec | null {
  return loadWidgets().find(w => w.id === id) ?? null;
}

export function isWidgetFeatureEnabled(): boolean {
  try {
    return !!localStorage.getItem('wm-widget-key');
  } catch {
    return false;
  }
}

export function getWidgetAgentKey(): string {
  try {
    return localStorage.getItem('wm-widget-key') ?? '';
  } catch {
    return '';
  }
}

function cleanSpanEntry(storageKey: string, panelId: string): void {
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return;
    const spans = JSON.parse(raw) as Record<string, number>;
    if (!(panelId in spans)) return;
    delete spans[panelId];
    if (Object.keys(spans).length === 0) {
      localStorage.removeItem(storageKey);
    } else {
      localStorage.setItem(storageKey, JSON.stringify(spans));
    }
  } catch {
    // ignore
  }
}
