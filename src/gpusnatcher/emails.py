from email.mime.text import MIMEText
from email.utils import formataddr
from smtplib import SMTP_SSL, SMTPResponseException

from gpusnatcher.logger import console


class EmailManager:
    """Class to manage email notifications."""

    def __init__(self, host_server: str, user: str, pwd: str, sender: str, receivers: list[str] | str) -> None:
        """Initialize the EmailManager.

        Args:
            host_server (str): The email server host.
            user (str): The email account username.
            pwd (str): The email account password.
            sender (str): The email sender address.
            receivers (list[str]): The list of email receiver addresses.
        """
        self.host_server = host_server
        self.user = user
        self.pwd = pwd
        self.sender = sender
        self.receivers = [receivers] if isinstance(receivers, str) else receivers

    def init_msg(self, subject: str, body: str) -> MIMEText:
        """Initialize the email message."""
        message = MIMEText(f"{body}", "plain", "utf-8")
        message["Subject"] = f"{subject}"
        message["From"] = formataddr(("GPUSnatcher", self.sender))
        message["To"] = ", ".join(self.receivers)
        return message

    def send_email(self, subject: str, body: str) -> None:
        """Send an email notification."""
        try:
            msg = self.init_msg(subject, body)
            with SMTP_SSL(self.host_server) as smtp:
                smtp.login(self.user, self.pwd)
                smtp.send_message(msg)
        except SMTPResponseException as e:
            if e.smtp_code == -1:
                pass
            else:
                raise
        except Exception as e:
            console.print(f"[red]Failed to send email: {e}[/red]")
            console.print_exception()
