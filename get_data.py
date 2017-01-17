import os
import imdb
import imdb.helpers
import urllib
import json
from multiprocessing import Pool
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, imdb.Person.Person) or isinstance(o, imdb.Company.Company):
            return {k: o[k] for k in o.keys()}
        elif isinstance(o, imdb.Movie.Movie):
            import ipdb; ipdb.set_trace() # BREAKPOINT
            return str(o.movieID)
        else:
            return super(PersonEncoder, self).default(o)


def save_movie(imdbId):
    json_path = 'mmimdb/dataset/' + imdbId + '.json'
    jpeg_path = 'mmimdb/dataset/' + imdbId + '.jpeg'
    try:
        if os.path.isfile(json_path) and os.path.isfile(jpeg_path):
            return
        ia = imdb.IMDb()
        m = ia.get_movie(imdbId)
        jsonData = {}
        for k in m.keys():
            v = m[k]
            jsonData[k] = v
        with open(json_path, 'w') as f:
            json.dump(jsonData, f, cls=PersonEncoder, indent=4)
        imgUrl = imdb.helpers.fullSizeCoverURL(m)
        if imgUrl is not None:
            imageData = urllib.urlopen(imdb.helpers.fullSizeCoverURL(m)).read()
            with open(jpeg_path, 'wb') as f:
                f.write(imageData)
        else:
            logger.log(logging.WARNING, imdbId + ':Does not have image poster')
    except Exception as e:
        logger.log(logging.ERROR, imdbId + ':' + str(e))
        if os.path.isfile(json_path):
            os.remove(json_path)
        if os.path.isfile(jpeg_path):
            os.remove(jpeg_path)

if __name__ == "__main__":
    with open('links.csv', 'r') as f:
        lines = f.readlines()

    imdbIds = [line.split(',')[1] for line in lines]
    p = Pool(8)
    p.map(save_movie, imdbIds[1:])
    p.close()
    # for imdbId in imdbIds[1:]:
    #    save_movie(imdbId)
