"""Create an administrator with an interactive password prompt."""
import getpass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()
from src.database.models import get_database_manager
from src.database.user_repository import UserRepository
from src.auth.security import DEFAULT_ROLE_PERMISSIONS

def main():
    manager = get_database_manager()
    manager.create_tables()
    repository = UserRepository(manager)
    username = input('Administrator username: ').strip()
    if not username or repository.get_user_by_username(username):
        raise SystemExit('Username is empty or already exists; no account changed.')
    password = getpass.getpass('Password (12+ characters): ')
    if len(password) < 12 or len(password.encode('utf-8')) > 72:
        raise SystemExit('Use 12+ characters and at most 72 UTF-8 bytes.')
    if password != getpass.getpass('Confirm password: '):
        raise SystemExit('Passwords do not match.')
    repository.create_user(username=username, password=password, role='admin',
                           permissions=DEFAULT_ROLE_PERMISSIONS['admin'])
    print('Administrator created. Start the API and open /login.html.')

if __name__ == '__main__':
    main()
