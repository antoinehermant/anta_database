/**
 * Email verification with the portal (portal/app.py): a code sent to the address, exchanged for
 * a session token kept in the browser, shared by the submit and contact pages.
 *
 * The pages have the same elements: #send-code, #code-field (with #code and #check-code) and
 * #verify-status.
 */

const KEY = "portal-verification";
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

interface Session {
  email: string;
  token: string;
  expires: number;
}

export const normalEmail = (email: string) => email.trim().toLowerCase();
export const validEmail = (email: string) => /^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email.trim());

function session(): Session | null {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "null");
  } catch {
    return null;
  }
}

/** The session token of a verified address, or "" when not verified (or expired). */
export function token(email: string): string {
  const s = session();
  return s && s.email === normalEmail(email) && s.expires > Date.now() ? s.token : "";
}

export const verified = (email: string) => token(email) !== "";

async function post(portalUrl: string, path: string, body: unknown) {
  const response = await fetch(`${portalUrl}/api/${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : `${response.status} ${response.statusText}`);
  return result;
}

const PROMPT = "We send you a code to verify your email address.";

/**
 * Bind the verification elements of a page. email() gives the address of the form, website()
 * the hidden field against bots; changed() is called once verified.
 */
export function setupVerification(portalUrl: string, email: () => string, website: () => string, changed: () => void) {
  const status = $("verify-status");
  const prompt = status.textContent || PROMPT;

  async function sendCode() {
    const address = normalEmail(email());
    if (!validEmail(address)) {
      status.textContent = "Enter your email address first.";
      return;
    }
    status.textContent = "Sending...";
    try {
      await post(portalUrl, "verify", { email: address, website: website() });
      status.textContent = `A code was sent to ${address}: enter it below (this can take a few minutes. Check your spam folder if it does not arrive).`;
      $("code-field").hidden = false;
      $("code").focus();
    } catch (e) {
      status.textContent = `The code could not be sent: ${(e as Error).message}`;
    }
  }

  async function checkCode() {
    const address = normalEmail(email());
    try {
      const result = await post(portalUrl, "verify/check", { email: address, code: $<HTMLInputElement>("code").value.trim() });
      localStorage.setItem(KEY, JSON.stringify({ email: address, token: result.token, expires: Date.now() + result.hours * 3600 * 1000 }));
      $<HTMLInputElement>("code").value = "";
      changed();
    } catch (e) {
      status.textContent = (e as Error).message;
    }
  }

  $("send-code").addEventListener("click", sendCode);
  $("check-code").addEventListener("click", checkCode);
  $("code").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      checkCode();
    }
  });

  /** Show whether the address of the form is verified. */
  return function show() {
    const ok = verified(email());
    $<HTMLButtonElement>("send-code").hidden = ok;
    if (ok) {
      $("code-field").hidden = true;
      status.textContent = `✓ ${normalEmail(email())} is verified.`;
      status.className = "verified";
    } else if (status.className === "verified") {
      status.textContent = prompt;
      status.className = "muted";
    }
  };
}
