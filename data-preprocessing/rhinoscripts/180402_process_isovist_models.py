
import rhinoscriptsyntax as rs
import Rhino
import scriptcontext
import System.Drawing
import os

def main():
    #basepath = "C:\\Users\\ksteinfe\\Desktop\\TEMP" #Kyle's Desktop
    basepath = "G:\\TEMP" #Matt's Desktop
    #basepath = "C:\\Users\\Matt\\Desktop\\Temp" #Matt's Laptop

    max_flrs = 99 # maximum number of folders to open
    max_fils = 99 # maximum number of files to open in each folder

    folder_tic = time.clock()
    fdr_cnt = 0
    for root, dirs, files in walklevel(basepath):
        
        print ("{}\t {}".format(fdr_cnt,root))
        #if (root == basepath): continue
        
        fil_cnt = 0
        for full_filename in files:
            filename, file_extension = os.path.splitext(full_filename)
            if file_extension != ".3dm": continue

            file_tic = time.clock()
            filepath = os.path.join(root, full_filename)
            rs.DocumentModified(False)
            rs.Command('_-Open {} _Enter'.format('"'+filepath+'"'))
            t = round(time.clock()-file_tic)
            print(filename+"\ttime:\t"+str(t))

            zoom_viewer("ViewerBox",35)#set factor for how far you want to zoom out from viewer
            patch_first_object_on_layer("ViewerBox",(255,233,233))#set color here
            rs.LayerPrintWidth("ViewerBox", width=0.01)#set print width here
            rs.Command("PrintDisplay"+"s"+" _-Enter"+"o"+" _-Enter"+" _-Enter")#ensures lineweights are displaying
            capture_view_antialias(os.path.join(basepath,filename+"_antialias.png"), (1000,1000) )#changeimagesize
            #capture_view(os.path.join(basepath,filename+".png"), (500,500) )#changeimagesize
            width="1200" #pixel count for width
            height="1200" #pixel count for height
            savepath=basepath + "\\" + filename +".png"
            #rs.Command("-ViewCaptureToFile scaledrawing=yes w "+width+" h "+height+" "+savepath+" ")
            fil_cnt+=1
            if fil_cnt > max_fils: break

        fdr_cnt+=1
        if fdr_cnt > max_flrs: break
    
    t = round(time.clock()-folder_tic)
    print("TOTAL\ttime:\t"+str(t))


def zoom_viewer(layername,f):
        rs.CurrentLayer(layername)
        set_active_view("Perspective")
        view = rs.CurrentView()
        set_disp_mode("Rendered")
        rs.ViewProjection(view,2)
        rs.ViewCameraTarget(view,(-1*f,1*f,1*f),(0,0,0))
        rs.ViewCameraLens(view,30)
        #rhobjs = scriptcontext.doc.Objects.FindByLayer(layername)
        #rs.SelectObject(rhobjs[0])
        #rs.ZoomSelected()

def zoom_out():
        rs.Command("Zoom"+" _-Enter"+"O"+" _-Enter")#ZoomOut

def patch_first_object_on_layer(layername,color):
    rhobjs = scriptcontext.doc.Objects.FindByLayer(layername)
    surface=rs.AddPlanarSrf(rhobjs[0])
    material_index = rs.AddMaterialToObject(surface)# add new material and get its index
    rs.MaterialColor(material_index, color)# assign material color
    try:
        return surface
    except:
        return False


def set_active_view(viewportName):
    RhinoDocument = Rhino.RhinoDoc.ActiveDoc
    view = RhinoDocument.Views.Find(viewportName, False)
    if view is None: return
    RhinoDocument.Views.ActiveView = view
    return view # returns RhinoCommon view. for Rhinoscript, use rs.CurrentView()


# scale_fac is capture to save size ratio
def capture_view_antialias(filePath, save_size=1.0, scale_fac = 2.0 ): 
    RhinoDocument = Rhino.RhinoDoc.ActiveDoc
    view = RhinoDocument.Views.ActiveView    
    vp = view.ActiveViewport
    
    try:
        capture_size = System.Drawing.Size(save_size[0]*scale_fac,save_size[1]*scale_fac)
    except:
        save_size = vp.Size.Width*save_size,vp.Size.Height*save_size
        capture_size = System.Drawing.Size(save_size[0]*scale_fac,save_size[1]*scale_fac)
    
    capture = view.CaptureToBitmap( capture_size )
    capture = resize_bitmap(capture,save_size)
    capture.Save(filePath);

def capture_view(filePath, save_size=1.0):
    RhinoDocument = Rhino.RhinoDoc.ActiveDoc
    view = RhinoDocument.Views.ActiveView    
    vp = view.ActiveViewport
    
    try:
        capture_size = System.Drawing.Size(save_size[0],save_size[1])
    except:
        save_size = vp.Size.Width*save_size,vp.Size.Height*save_size
        capture_size = System.Drawing.Size(save_size[0],save_size[1])
    
    capture = view.CaptureToBitmap( capture_size )

    capture.Save(filePath);


def resize_bitmap(src_img,size):
    dest_rect = System.Drawing.Rectangle(0,0,size[0],size[1])
    dest_img = System.Drawing.Bitmap(size[0],size[1])
    dest_img.SetResolution(src_img.HorizontalResolution, src_img.VerticalResolution)
    
    with System.Drawing.Graphics.FromImage(dest_img) as g:
        g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy
        g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality
        g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic
        g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality
        g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality
        
        with System.Drawing.Imaging.ImageAttributes() as wrap_mode:
            wrap_mode.SetWrapMode(System.Drawing.Drawing2D.WrapMode.TileFlipXY)
            g.DrawImage(src_img, dest_rect, 0,0, src_img.Width, src_img.Height, System.Drawing.GraphicsUnit.Pixel, wrap_mode)
    return dest_img


def set_disp_mode(modename):
    desc = Rhino.Display.DisplayModeDescription.FindByName(modename)
    if desc: 
        scriptcontext.doc.Views.ActiveView.ActiveViewport.DisplayMode = desc
        #scriptcontext.doc.Views.Redraw()

import rhinoscriptsyntax as rs
import os, time

def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir), "Directory not found:\t{}".format(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]
            

if __name__ == "__main__":
    # execute only if run as a script
    main()