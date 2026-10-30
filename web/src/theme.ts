/**
 * Light or dark theme of the viewer and the download page: dark by default,
 * or the choice saved in localStorage ("Light", "Dark", or "Auto" for the mode
 * of the browser). The inline script in the <head> of each page sets
 * data-theme on <html> the same way before the first paint.
 */
export type Theme = "Auto" | "Light" | "Dark";

const darkMode = window.matchMedia("(prefers-color-scheme: dark)");
const listeners: (() => void)[] = [];

export function savedTheme(): Theme {
  const theme = localStorage.getItem("theme");
  return theme === "Light" || theme === "Auto" ? theme : "Dark";
}

export function isDark(theme: Theme = savedTheme()): boolean {
  return theme === "Dark" || (theme === "Auto" && darkMode.matches);
}

/** Save and apply a theme, and tell the listeners. */
export function setTheme(theme: Theme): void {
  localStorage.setItem("theme", theme);
  applyTheme();
}

function applyTheme(): void {
  document.documentElement.dataset.theme = isDark() ? "dark" : "light";
  for (const listener of listeners) listener();
}

/** Called after every change of the theme (also from the mode of the browser, in Auto). */
export function onThemeChange(listener: () => void): void {
  listeners.push(listener);
}
darkMode.addEventListener("change", applyTheme);

const SUN =
  '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4.5"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const MOON =
  '<svg width="18" height="18" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4 8.5 8.5 0 1 0 20 14.5Z"/></svg>';

/** A button switching between the light and the dark theme. */
export function themeButton(button: HTMLButtonElement): void {
  const update = () => {
    const dark = isDark();
    // The icon of the theme it switches to
    button.innerHTML = dark ? SUN : MOON;
    const label = dark ? "Light theme" : "Dark theme";
    button.setAttribute("aria-label", label);
    button.title = label;
  };
  button.addEventListener("click", () => setTheme(isDark() ? "Light" : "Dark"));
  onThemeChange(update);
  update();
}
