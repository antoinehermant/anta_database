/** Front page and privacy page: the light or dark theme button, and the texts that depend on
 * whether the submission portal is public (VITE_PORTAL_PUBLIC=true at the build). */
import { themeButton } from "./theme";

themeButton(document.querySelector<HTMLButtonElement>("#theme-toggle")!);

const portalPublic = import.meta.env.VITE_PORTAL_PUBLIC === "true";
for (const el of document.querySelectorAll<HTMLElement>("[data-portal]"))
  el.hidden = (el.dataset.portal === "open") !== portalPublic;
