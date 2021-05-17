from shutil import move
import os

def splitter(folder, train, test, valid):
    filenames = os.listdir(folder)

    for (i, fname) in enumerate(filenames):

        src = os.path.join(folder, fname)

        if (i % 5) == 0:
            # VALIDATION
            if (i % 10) == 0:
                    dst = os.path.join(valid + "/fake", fname)
            # TESTING
            else:
                    dst = os.path.join(test + "/fake", fname)
        # Training
        else:
                dst = os.path.join(train + "/fake", fname)

        move(src, dst)


valid = "faces/valid"
test = "faces/test"
train = "faces/train"

splitter("1m_faces_03", train, test, valid)
