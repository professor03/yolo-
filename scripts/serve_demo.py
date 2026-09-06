"""Loopback-only demo: personal account, isolated state, no camera workers."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['DISABLE_CAMERA_STREAMS'] = 'true'
os.environ['RTSP_CAMERA_URL'] = ''
os.environ['MJPG_CAMERA_URL'] = ''
os.environ['DATABASE_URL'] = 'sqlite:///data/public-demo.db'
os.environ['DATASTORE_PATH'] = 'data/public-demo.json'
if __name__ == '__main__':
    from src.database.models import get_database_manager
    from src.database.user_repository import UserRepository
    get_database_manager().create_tables()
    if not UserRepository().list_users():
        from create_admin import main
        main()
    import uvicorn
    print('Local demo: http://127.0.0.1:8000/login.html')
    uvicorn.run('src.server.app:app', host='127.0.0.1', port=8000)
