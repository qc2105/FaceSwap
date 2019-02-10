import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import spatial

#prawe oko - 53, 56
#lewe oko - 23, 20
#prawa brew - 48, 49, 50
#lewa brew - 17, 16, 15
#usta zewnetrzne 64, 7, 31, 8, 79, 80
#usta wewnetrzne 89, 87, 88, 40
#broda 65, 10, 32
#broda2 62, 61, 63, 29, 28, 30
idxs = [26, 59, 5, 6, 94, 111, 112,
        53, 98, 104, 56, 110, 100, 
        23, 103, 97, 20, 99, 109,      
        48, 49, 50,     
        17, 16, 15,
        64, 7, 31, 8, 79, 80, 85, 86,
        89, 87, 88, 40,
        65, 10, 32,
        62, 61, 63, 29, 28, 30]      
#prawe oko - 36, 39
#lewe oko - 42, 45
#prawa brew - 17, 19, 21 
#lewa brew - 22, 24, 26
#usta zewnetrzne 48, 51, 54, 57, 53, 49
#ustwa wewnetrzne 60, 62, 64, 66
#broda 7, 8, 9
idxs2 = [35, 31, 30, 33, 28, 34, 32,
            36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47, 
            17, 19, 21, 
            22, 24, 26,
            48, 51, 54, 57, 53, 49, 55, 59,
            60, 62, 64, 66,
            7, 8, 9,
            0, 2, 4, 16, 14, 12]

def saveAsRaw(mean3, modes3, vertices, outDir="../output/"):
    np.savetxt(outDir + "mean3.dim", mean3.shape)
    np.savetxt(outDir + "mean3.txt", mean3)

    for i in range(3):
        temp = modes3[:, :, i]
        np.savetxt(outDir + "modes3" + str(i) + ".dim", temp.shape)
        np.savetxt(outDir + "modes3" + str(i) + ".txt", temp)

    np.savetxt(outDir + "vertices.dim", vertices.shape)
    np.savetxt(outDir + "vertices.txt", vertices)

def saveOBJ(name, verts, faces):
    f = open(name, "w+")

    for vert in verts:
        s = "v "
        s += str(vert[0]) + " "
        s += str(vert[1]) + " "
        s += str(vert[2]) + "\n"

        f.write(s)

    n = 0
    for face in faces:
        s = "f "
        for i in range(face.shape[0] - 1):
            s += str(face[i] + 1) + "//" + str(n + 1) + " "
            n += 1

        s += str(face[-1] + 1) + "//" + str(n + 1) + "\n"
        n += 1

        f.write(s)
        
    f.close()

def loadOBJ(name):
    f = open(name, "r")
    verts = []
    faces = []

    lines = f.readlines()
    for line in lines:
        if lines[0] == '#':
            continue

        pieces = line.split()
        if line[0] == 'v':
            verts.append([float(pieces[i]) for i in range(1, len(pieces))])
        if line[0] == 'f':
            faces.append([int(pieces[i]) - 1 for i in range(1, len(pieces))])
        
    f.close()

    verts = np.array(verts)
    faces = np.array(faces)
    return verts, faces

def addBackgroundPoints(mean3, modes3, faces, size, shape=(6, 4)):
    P = np.zeros((2, 3))
    P[0, 0] = 1
    P[1, 1] = 1
    shape2D = np.dot(mean3, P.T)

    lowBounds = np.min(shape2D * size, axis=0)
    highBounds = np.max(shape2D * size, axis=0)

    leftBackPoints = np.zeros((shape[0], 2))
    rightBackPoints = np.zeros((shape[0], 2))
    topBackPoints = np.zeros((shape[1], 2))
    bottomBackPoints = np.zeros((shape[1], 2))

    leftBackPoints[:, 0] = lowBounds[0]
    leftBackPoints[:, 1] = np.linspace(lowBounds[1], highBounds[1], shape[0])
    rightBackPoints[:, 0] = highBounds[0]
    rightBackPoints[:, 1] = np.linspace(lowBounds[1], highBounds[1], shape[0])

    topBackPoints[:, 0] = np.linspace(lowBounds[0], highBounds[0], shape[1])
    topBackPoints[:, 1] = highBounds[1]   
    bottomBackPoints[:, 0] = np.linspace(lowBounds[0], highBounds[0], shape[1]) 
    bottomBackPoints[:, 1] = lowBounds[1]

    backPoints = np.vstack((leftBackPoints, rightBackPoints, topBackPoints, bottomBackPoints))
    allPoints2D = np.vstack((shape2D, backPoints))

    delaunay = spatial.Delaunay(shape2D, incremental=True)
    delaunay.add_points(backPoints, False)

    backIndices = range(shape2D.shape[0], allPoints2D.shape[0])
    backFaces = []
    for i in range(delaunay.vertices.shape[0]):
        temp = [x in backIndices for x in delaunay.vertices[i]]
        if any(temp):
            backFaces.append(delaunay.vertices[i])

    #vertices of all faces
    allFaces = np.vstack((faces, backFaces))

    backgroundZ = np.min(mean3, 0)[2]
    backPoints = np.hstack((backPoints, np.zeros((backPoints.shape[0], 1))))
    backPoints[:, 2] = backgroundZ

    allMean3 = np.vstack((mean3, backPoints))
    allModes3 = np.zeros((modes3.shape[0], allMean3.shape[0], 3))

    for i in range(modes3.shape[0]):
        allModes3[i, :modes3.shape[1]] = modes3[i]

    #plots.plotPoints(allPoints2D)
    #plots.plotMesh(allPoints2D, allFaces)
    #plt.show()

    return allMean3, allModes3, allFaces

f = open('../candide3.wfm.txt','r')

lines = []
for line in f:
    lines.append(line.strip())

i = 0

mean3 = None
modes3 = None
vertices =None
while True:
    if "VERTEX LIST" in lines[i]:
        nVerts = int(lines[i+1])
        tempList = lines[i+2:i+nVerts+2]
        tempList = [map(float, x.split()) for x in tempList]
        mean3 = np.array(tempList)
        i += nVerts + 1

    if "FACE LIST" in lines[i]:
        nVerts = int(lines[i+1])
        tempList = lines[i+2:i+nVerts+2]
        tempList = [map(int, x.split()) for x in tempList]
        vertices = np.array(tempList)
        i += nVerts + 1


    # or "Nose z-extension" in lines[i]
    if ("# AUV" in lines[i]) and "lid" not in lines[i]: # or "Cheeks z" in lines[i])
        print lines[i]
        nVerts = int(lines[i+1])
        tempList = lines[i+2:i+nVerts+2]
        tempList = [map(float, x.split()) for x in tempList]
        tempArr = np.array(tempList)

        tempMode = np.zeros((1, mean3.shape[0], mean3.shape[1]))
        tempMode[0, tempArr[:, 0].astype(np.int32)] = tempArr[:, 1:]

        if modes3 == None:
            modes3 = tempMode
        else:
            modes3 = np.vstack((modes3, tempMode))

    if ("r_m_eyebrow" in lines[i] or "l_m_eyebrow" in lines[i]):#"strecth_l_nose" in lines[i] or "strecth_r_nose" in lines[i]:
        print lines[i]
        i += 1
        nVerts = int(lines[i+1])
        tempList = lines[i+2:i+nVerts+2]
        tempList = [map(float, x.split()) for x in tempList]
        tempArr = np.array(tempList)

        tempMode = np.zeros((1, mean3.shape[0], mean3.shape[1]))
        tempMode[0, tempArr[:, 0].astype(np.int32)] = tempArr[:, 1:]

        if modes3 == None:
            modes3 = tempMode
        else:
            modes3 = np.vstack((modes3, tempMode))
        
        i += nVerts + 1

    if "END OF" in lines[i]:
        break

    i += 1


#deepening of the face
#indsToDeepen = [63, 61, 62, 47, 45, 44, 0,
#                11, 12, 14, 29, 28, 30]
#mean3[indsToDeepen, 2] -= 0.35

mouthVertices = [[82, 89, 84], [82, 87, 40], [82, 84, 40], [40, 81, 87], [81, 40, 83], [81, 83, 88]]
vertices = np.vstack((vertices, mouthVertices))

r = np.array([0, 0, 3.14])
R = cv2.Rodrigues(r)[0]
for i in range(modes3.shape[0]):
    modes3[i] = np.dot(modes3[i], R.T)
mean3 = np.dot(mean3, R.T)

R = np.zeros((3, 3))
R[0][0] = -1
R[1][1] = 1
R[2][2] = -1
for i in range(modes3.shape[0]):
    modes3[i] = np.dot(modes3[i], R.T)
mean3 = np.dot(mean3, R.T)

mean3 = mean3.T
modes3 = np.transpose(modes3, [0, 2, 1])
#plots.plot3D(mean3)
#plt.show()

#_, vertices = loadOBJ("../candideInterior3.obj")
np.savez("../candide.npz", mean3DShape=mean3, blendshapes=modes3, mesh=vertices, idxs3D=idxs, idxs2D=idxs2)
saveOBJ("../candide.obj", mean3.T, vertices)

#P = np.zeros((2, 3))
#P[0, 0] = 1
#P[1, 1] = 1
#shape2D = np.dot(mean3 + 1 * modes3[1], P.T)

#ax = plt.gca(aspect="equal")
#ax.invert_yaxis()
#plots.plotPoints(shape2D, ax, True)
#plots.plotMesh(shape2D, vertices, ax)
#plt.show()