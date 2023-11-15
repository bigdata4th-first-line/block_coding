import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

import urllib.request


def get_image(img_name):
    try:
        cred = credentials.Certificate('service-key.json')
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://firstline-17bda-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })
    except:
        print('The default Firebase app already exists.')
        pass

    docs = db.reference()
    data = docs.get()
    url = data[img_name]['image_url']

    urllib.request.urlretrieve(url, "./images/" + img_name + ".jpg")