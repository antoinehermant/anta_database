/**
 * The page a closed private preview of a submission sends to (src/preview.ts), with the
 * status of the submission (portal/manage.py) in ?status=.
 */
import { themeButton } from "./theme";

const $ = (id: string) => document.getElementById(id)!;
themeButton($("theme-toggle") as HTMLButtonElement);

const status = new URLSearchParams(location.search).get("status");
if (status === "accepted") {
  $("message").innerHTML =
    'Your dataset is now part of the AntADatabase: see it in the <a href="/viewer/">3D viewer</a>, or download it from ' +
    'the <a href="/download/">download page</a>. Its private preview is now closed. Please check your emails for the details.';
} else if (status === "failed") {
  $("title").textContent = "Your submission could not be compiled";
  $("message").textContent =
    "Your dataset could not be compiled automatically, so there is no 3D preview. We were told, and will come back to you: please check your emails.";
} else if (status === "missing") {
  $("title").textContent = "No such preview";
  $("message").textContent =
    "This preview does not exist: please check the link in your email. Previews are also closed once the review of the submission is done.";
}
