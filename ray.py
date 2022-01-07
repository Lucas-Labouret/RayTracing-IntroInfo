from math import sqrt
from csv import DictReader
from sys import argv, exit
import corrige_ray

## Question 0

NOM = "LABOURET"
PRENOM = "LUCAS"
EMAIL = "LUCAS.LABOURET@UNIVERSITE-PARIS-SACLAY.FR"

### Opérations sur les vecteurs
### (section 4.1 de l'énoncé)
### Un vecteur est simplement représenté par un couple (x, y)

## Question 1

def add(v1, v2):
    """Addition de deux vecteurs"""
    return tuple(v1[i]+v2[i] for i in range(len(v1)))

def sub(v1, v2):
    """Différence de deux vecteurs"""
    return tuple(v1[i]-v2[i] for i in range(len(v1)))

def mul(k, v):
    """Multiplication par une constante"""
    return tuple(k*coord for coord in v)

def dot(v1, v2):
    """Produit scalaire de deux vecteurs"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i]*v2[i]
    return sum

def norm(v):
    """Norme d'un vecteur """
    return sqrt(dot(v, v))

def normalize(v):
    return mul(1/norm(v), v)

## Question 2

def test_add():
    """Fonction de test pour add"""
    assert add((1,0),(1,1)) == (2,1)
    assert add((1,0,2),(1,1,17)) == (2,1,19)


test_add()

def test_sub():
    """Fonction de test pour sub"""
    assert sub((1,0),(1,1)) == (0,-1)
    assert sub((1,0,2),(1,1,17)) == (0,-1,-15)

test_sub()

def test_mul():
    """Fonction de test pour mul"""
    assert mul(-1,(1,0)) == (-1,0)
    assert mul(4,(1,1,17)) == (4,4,68)

test_mul()

def test_dot():
    """Fonction de test pour dot"""
    assert dot((1,0),(1,1)) == 1
    assert dot((1,0,2),(1,1,17)) == 35

test_dot()

def test_norm():
    """Fonction de test pour norm"""
    assert 1 - 10**-10 < norm((1,0)) < 1 + 10**-10
    assert sqrt(5) - 10**-10 < norm((1,0,2)) < sqrt(5) + 10**-10

test_norm()

def test_normalize():
    """Fonction de test pour normalize"""
    ##À COMPLÉTER

test_normalize()



### Opérations sur les images
### (section 4.2 de l'énoncé)
### Une image est représentée par un triplet (i, w, h) où :
### - i est un bytearray de taille w * h * 3
### - w est la largeur de l'image (en pixels)
### - h est la hauteur de l'image (en pixels)

## Question 3
def init_image(w, h):
    """Initialise une image de w pixels de large et h pixel de haut"""
    return (bytearray(w*h*3), w, h)

## Question 4
def set_pixel(img, x, y, c):
    """Met le pixel au coordonnées (x, y) à la couleur c. C'est un est
    triplet (r, v, b) de valeurs. Les valeurs supérieures à 1 (resp. inférieures à
    0) sont mises à 1 (resp. 0).
    """
    ar, width, height = img
    pos = 3*y*width + 3*x
    for i in [0, 1, 2]:
        ar[pos+i] = max(0, min(255, int(c[i]*255)))
    return img

### Fonction donnée, ne pas modifier
def save_image(chemin, img):
    """Écrit l'image img dans le fichier dont le chemin est donné. Si
    le fichier existe, il est supprimé. L'image est stockée au format PPM"""
    buff, w, h = img
    with open(chemin, "wb") as f:
        f.write(b'P6\n')
        f.write(str(w).encode())
        f.write(b' ')
        f.write(str(h).encode())
        f.write(b'\n255\n')
        f.write(buff)

## Question 5
def test_img():
    """Test des fonctions set_pixel et init_image"""

    #Crée une image noire de 100x100
    save_image("black100.ppm", init_image(100, 100))

    #Crée un cercle rouge centré de rayon 50 sur une image blanche de 200x200
    img = init_image(200, 200)
    for y in range(200):
        for x in range(200):
            if 49 < sqrt((x-100)**2 + (y-100)**2) < 51:
                img = set_pixel(img, x, y, (1,0,0))
            else:
                img = set_pixel(img, x, y, (1,1,1))
    save_image("testCircle.ppm", img)

### Attention, si vous faites une génération d'image un peu coûteuse, vous pouvez
### commenter l'appel ci-dessous après avoir testé

#test_img()

### Fonctions de ray tracing
### Section 5 de l'énoncé

## Question 6
def pixel_to_point(w, h, xmin, xmax, ymin, ymax, px, py):
    """Convertit un pixel (px, py) en un point du plan."""
    x = px*(xmax-xmin)/w + xmin
    y = py*(ymax-ymin)/h + ymin
    return x, y

## Question 7
def sphere_intersect(c, r, v, d):
    """Calcule l'intersection entre une sphere de centre c (vecteur) et de rayon r
    (flottant) et une droite passant par v (vecteur) et de direction d (vecteur)
    """
    return corrige_ray.sphere_intersect(c, r, v, d)
    b = 2*dot(d, sub(v, c))
    delta = b**2 - 4*((norm(sub(v, c))**2) - (r**2))
    if delta < 0:
        return None

    k2 = -b-sqrt(delta)/2
    if k2 < 0:
        return None

    return k2

INF = float('inf')

## Question 8
def nearest_intersection(objs, o, d):
    """Renvoie la sphère la plus proche qui intersecte la droite partant de o dans
    la direction d, ainsi que la distance d'intersection depuis o. S'il n'y a pas
    d'intersection, renvoie (None, INF)"""
    min_dist = INF
    min_obj = None
    for curr_obj in objs:
        curr_dist = sphere_intersect(curr_obj["center"], curr_obj["radius"], o, d)
        if curr_dist == None:
            continue
        if curr_dist < min_dist:
            min_dist = curr_dist
            min_obj = curr_obj
    return min_obj, min_dist

## Question 9
def compute_color(obj, v, n, l):
    """calcule la couleur du point v se trouvant à la surface de l'objet obj.
    n est le vecteur normal au point d'intersection et l le vecteur unitaire dans
    la direction de la source de lumière.
    """
    a = obj["ambiant"]
    d = mul(dot(l,n), obj["diffuse"])
    s = mul(abs(dot(n, normalize(add(l,v))))**(obj["shininess"]/4), obj["specular"])
    return add(a, add(d, s))


## NE PAS MODIFIER LE CODE CI-DESSOUS
def trace(w, h, xmin, xmax, ymin, ymax, camera, light,objs):

    img = init_image(w, h)

    for py in range(h):
        for px in range(w):
            x, y = pixel_to_point(w, h, xmin, xmax, ymin, ymax, px, py)
            p = (x, y, 0)
            vp = sub(p, camera)
            d = normalize(vp)

            obj, dist = nearest_intersection(objs, camera, d)

            if obj is None:
                couleur = (0, 0, 0)
                set_pixel(img, px, h - py - 1, couleur)
                continue

            x_point = add(camera, mul(dist, d))
            l = normalize(sub(light, x_point))
            obstacle, dist_obst = nearest_intersection(objs, x_point, l)
            if dist_obst < norm(sub(light, x_point)):
                couleur = (0, 0, 0)
            else:
                n = normalize (sub(x_point, obj["center"]))
                couleur = compute_color(obj, camera, n, l)
            set_pixel(img, px, h - py - 1, couleur)

        print("-", end="", flush=True)
    print("\n")
    return img



### Lecture de description de scene
###

def read_vector(s):
    fields = s.split(",")
    if len(fields) != 3:
        raise ValueError("Erreur de chargement")
    return [ float (n) for n in fields ]

def load_scene (chemin):
    """Charge un fichier de description de scène. En cas d'erreur, la fonction
    lève une exception 'Exception("Erreur de chargement")'
    """
    try:
        with open (chemin, "r") as f:
            w = int (f.readline())
            h = int (f.readline())
            xmin = float(f.readline())
            xmax = float(f.readline())
            ymin = float(f.readline())
            ymax = float(f.readline())
            camera = read_vector(f.readline())
            light = read_vector(f.readline())
            objects = list(DictReader(f, delimiter=";"))
            for obj in objects:
                obj['center'] = read_vector(obj['center'])
                obj['radius'] = float(obj['radius'])
                obj['ambiant'] = read_vector(obj['ambiant'])
                obj['diffuse'] = read_vector(obj['diffuse'])
                obj['specular'] = read_vector(obj['specular'])
                obj['shininess'] = min(100, max(0, float(obj['shininess'])))
                obj['reflection'] = min(1, max(0, float(obj['reflection'])))
            return (w, h, xmin, xmax, ymin, ymax, camera, light, objects)
    except:
        raise ValueError("Erreur de chargement")

def usage():
    print(f"Usage: {argv[0]} <fichier.scene>")
    exit (1)

if __name__ == "__main__":
    if len (argv) != 2:
        usage()

    fichier = argv[1]
    if len(fichier) < 6 or fichier[-6:] != ".scene":
        usage()

    out = fichier[0:-6] + ".ppm"

    w, h, xmin, xmax, ymin, ymax, camera, lum, objs = load_scene(fichier)
    img = trace(w, h, xmin, xmax, ymin, ymax, camera, lum, objs)
    save_image(out, img)
