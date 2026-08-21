import ee
from pathlib import Path

# 1. Destroy the old cached credentials so GEE has no memory of your old project
cred_path = Path.home() / '.config' / 'earthengine' / 'credentials'
if cred_path.exists():
    cred_path.unlink()
    print("✅ Wiped old cached credentials from hard drive.") #

# 2. Force Notebook Mode (This forces the browser to ask for your Cloud Project)
print("🔑 Opening browser for mandatory project selection...")
ee.Authenticate(auth_mode='notebook', force=True) #

# 3. Test Initialization 
NEW_PROJECT_ID = 'ee-one-third-nine-to-five-v6'
ee.Initialize(project=NEW_PROJECT_ID) #
print(f"🚀 SUCCESS! Earth Engine is fully locked to project: {NEW_PROJECT_ID}")