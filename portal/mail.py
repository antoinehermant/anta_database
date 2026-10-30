"""Outgoing emails of the portal, by SMTP, or to the log when no SMTP server is set up."""

import logging
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr, make_msgid

from . import config

log = logging.getLogger("portal.mail")


def send(to: str, subject: str, body: str, reply_to: str = "", attachments: list[tuple[str, bytes, str]] = ()) -> None:
    """An email, with attachments as (file name, content, MIME type)."""
    message = EmailMessage()
    message["From"] = formataddr(("AntADatabase", config.SMTP_FROM or "noreply@antadatabase.com"))
    message["To"] = to
    message["Subject"] = subject
    message["Message-ID"] = make_msgid(domain="antadatabase.com")
    if reply_to:
        message["Reply-To"] = reply_to
    message.set_content(body)
    for name, content, mime in attachments:
        maintype, subtype = mime.split("/", 1)
        message.add_attachment(content, maintype=maintype, subtype=subtype, filename=name)

    if not config.SMTP_HOST:
        names = ", ".join(name for name, _, _ in attachments)
        log.info("Email (no SMTP_HOST, not sent) to %s: %s\n%s%s", to, subject, body, f"\n[attached: {names}]" if names else "")
        return
    if config.SMTP_SECURITY == "ssl":
        with smtplib.SMTP_SSL(config.SMTP_HOST, config.SMTP_PORT, context=ssl.create_default_context(), timeout=30) as smtp:
            _send(smtp, message)
    else:
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=30) as smtp:
            if config.SMTP_SECURITY == "starttls":
                smtp.starttls(context=ssl.create_default_context())
            _send(smtp, message)
    log.info("Email sent to %s: %s", to, subject)


def _send(smtp: smtplib.SMTP, message: EmailMessage) -> None:
    if config.SMTP_USER:
        smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
    smtp.send_message(message)


def to_maintainer(subject: str, body: str, reply_to: str = "", attachments: list[tuple[str, bytes, str]] = ()) -> None:
    if not config.MAINTAINER_EMAIL:
        log.warning("No PORTAL_MAINTAINER_EMAIL: %s\n%s", subject, body)
        return
    send(config.MAINTAINER_EMAIL, subject, body, reply_to, attachments)


if __name__ == "__main__":
    # Test of the settings: python -m portal.mail sends an email to the maintainer
    logging.basicConfig(level=logging.INFO)
    to_maintainer("AntADatabase portal: test email", f"The portal can send emails ({config.SMTP_HOST or 'log only'}).\n")
