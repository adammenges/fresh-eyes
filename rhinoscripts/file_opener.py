import rhinoscriptsyntax as rs
import os

basepath = "W:\\Box Sync\\Berkeley\\Courses Active\\100d s18\\10-15-100 Houses\\10-15-100 Massings"
folderpath = os.path.join(basepath,"XX - Test")

for filename in os.listdir(folderpath):
    if filename.endswith(".3dm"):
        filepath = os.path.join(folderpath, filename)
        print(filepath)
        rs.DocumentModified(False)
        rs.Command('_-Open {} _Enter'.format('"'+filepath+'"'))
        

