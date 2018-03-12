import rhinoscriptsyntax as rs
import os, time

basepath = "W:\\Box Sync\\Berkeley\\Courses Active\\100d s18\\10-15-100 Houses\\10-15-100 Massings"
folderpath = os.path.join(basepath,"AA - Cylindrical")

folder_tic = time.clock()
for filename in os.listdir(folderpath):
    if filename.endswith(".3dm"):
        file_tic = time.clock()
        filepath = os.path.join(folderpath, filename)
        rs.DocumentModified(False)
        rs.Command('_-Open {} _Enter'.format('"'+filepath+'"'))
        t = round(time.clock()-file_tic)
        print(filename+"\ttime:\t"+str(t))

t = round(time.clock()-folder_tic)
print("TOTAL\ttime:\t"+str(t))