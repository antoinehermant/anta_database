/**
 * Contact page: a message sent to the maintainer by the portal (portal/app.py, /api/contact).
 * Without the portal (a build without VITE_PORTAL_URL), it falls back to an email.
 */
import { themeButton } from "./theme";
import { setupVerification, token, validEmail, verified } from "./verify";

const PORTAL_URL = (import.meta.env.VITE_PORTAL_URL ?? "").replace(/\/$/, "");
const CONTACT_EMAIL = "antoine.hermant@unibe.ch";
// Files, as in the portal (portal/config.py): stored with the message, from a verified address
const MAX_FILES = 10;
const MAX_MB = 500;
const EXTENSIONS = [".csv", ".tab", ".txt", ".zip", ".pdf", ".png", ".jpg", ".jpeg"];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const field = (id: string) => $<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>(id);

themeButton($<HTMLButtonElement>("theme-toggle"));

// Name and email from the submission form, if filled in there
try {
  const draft = JSON.parse(localStorage.getItem("submit-draft") ?? "null");
  field("name").value = draft?.contributor?.name ?? "";
  field("email").value = draft?.contributor?.email ?? "";
} catch {
  // No draft
}
// Sending files needs a verified email address
const chosenFiles = () => [...($<HTMLInputElement>("files").files ?? [])];
const showVerification = setupVerification(
  PORTAL_URL,
  () => field("email").value,
  () => field("website").value,
  () => showVerification(),
);
function updateVerification() {
  $("verification").hidden = chosenFiles().length === 0;
  showVerification();
}
$("files").addEventListener("change", updateVerification);
$("email").addEventListener("input", updateVerification);
updateVerification();

const topic = new URLSearchParams(location.search).get("topic");
if (topic && [...$<HTMLSelectElement>("topic").options].some((o) => o.value === topic)) field("topic").value = topic;

$("contact-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const status = $("status");
  const button = $<HTMLButtonElement>("send");
  const name = field("name").value.trim();
  const email = field("email").value.trim();
  const message = field("message").value.trim();
  if (!name || !validEmail(email) || message.length < 10) {
    status.textContent = "Please give your name, a valid email address and a message.";
    return;
  }
  const files = chosenFiles();
  const size = files.reduce((n, f) => n + f.size, 0);
  const wrongType = files.find((f) => !EXTENSIONS.some((e) => f.name.toLowerCase().endsWith(e)));
  if (files.length > MAX_FILES || size > MAX_MB * 1024 * 1024 || wrongType) {
    status.textContent = wrongType
      ? `${wrongType.name}: only ${EXTENSIONS.join(", ")} files.`
      : `Limit exceeded: at most ${MAX_FILES} files and ${MAX_MB} MB in total. Try to reduce the size of your files.`;
    return;
  }
  if (files.length && !verified(email)) {
    status.textContent = "To send files, please verify your email address first (below the files).";
    return;
  }
  if (!import.meta.env.DEV && !PORTAL_URL) {
    // No portal: write the email instead
    const subject = encodeURIComponent(`AntADatabase: ${$<HTMLSelectElement>("topic").selectedOptions[0].text}`);
    location.href = `mailto:${CONTACT_EMAIL}?subject=${subject}&body=${encodeURIComponent(message)}`;
    return;
  }
  button.disabled = true;
  status.textContent = "Sending...";
  try {
    const body = new FormData();
    body.append("name", name);
    body.append("email", email);
    body.append("topic", field("topic").value);
    body.append("message", message);
    body.append("website", field("website").value);
    for (const f of files) body.append("files", f, f.name);
    const response = await fetch(`${PORTAL_URL}/api/contact`, {
      method: "POST",
      body,
      headers: files.length ? { Authorization: `Bearer ${token(email)}` } : {},
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : response.statusText);
    $("sent-email").textContent = email;
    $("contact-form").hidden = true;
    $("sent").hidden = false;
  } catch (e) {
    status.textContent = `The message could not be sent (${(e as Error).message}): please email ${CONTACT_EMAIL}.`;
    button.disabled = false;
  }
});
