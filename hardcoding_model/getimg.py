import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import urllib.request


def get_image():
    try:
        cred = credentials.Certificate('service-key.json')
        firebase_admin.initialize_app(cred)
    except:
        print('The default Firebase app already exists.')
        pass

    db = firestore.client()

    docs = db.collection('user').get()
    image_dict = docs[-1].to_dict()

    url = image_dict['image']
    img_name = url.split('F')[1].split('?')[0]

    urllib.request.urlretrieve(url, "./images/" + img_name + ".jpg")
    
    return img_name
