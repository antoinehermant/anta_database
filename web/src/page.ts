/** Front page and submit page: only the light or dark theme button. */
import { themeButton } from "./theme";

themeButton(document.querySelector<HTMLButtonElement>("#theme-toggle")!);
