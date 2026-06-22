"""
GestureCanvas v10 — Ultimate Vector-Grade Edition
- Professional Mask-Based Fill: Generates hard binary mask, dilates, and fills UNDER the line.
- Pre-Multiplied Alpha Compositing: Original anti-aliased strokes remain 100% solid.
- Localized Blending: Prevents other shapes from fading or turning transparent.
- Zero Gaps, Zero Halos, Zero Leaks.
"""
import cv2
import numpy as np
import math
import time
import HandTrackerModule as htm
from final_shape_snap import snap_from_stroke, SnapAnimation, LayerManager

# ── CONFIG ───────────────────────────────────────────────────
HEADER_HEIGHT          = 156
CAM_WIDTH              = 1280
CAM_HEIGHT             = 720
SCALE                  = CAM_WIDTH / 1600
MIN_BRUSH, MAX_BRUSH   = 4,  50
MIN_ERASER,MAX_ERASER  = 20, 120
COLOR_Y_START          = 19
COLOR_Y_END            = 65
HIGHLIGHT_COLOR        = (80,  60,  180)
HOVER_COLOR            = (70,  70,  70)
PANEL_CURSOR_COLOR     = (40,  40,  40)
AI_RESULT_FRAMES       = 220
RESIZE_DEBOUNCE        = 12
RESIZE_LOCK            = 20
HOVER_DEBOUNCE         = 5
CLICK_COOLDOWN         = 20   
FIST_CONFIRM           = 8    

TOOL_ZONES = {
    "colors":   (int(0    * SCALE), int(468  * SCALE)),
    "brush":    (int(468  * SCALE), int(630  * SCALE)),
    "eraser":   (int(630  * SCALE), int(790  * SCALE)),
    "undo":     (int(790  * SCALE), int(945  * SCALE)),
    "clear":    (int(945  * SCALE), int(1100 * SCALE)),
    "save":     (int(1100 * SCALE), int(1260 * SCALE)),
    "ai":       (int(1260 * SCALE), int(1415 * SCALE)),
    "settings": (int(1415 * SCALE), int(1600 * SCALE)),
}
COLOR_ZONES = [
    ((int(20  * SCALE), int(62  * SCALE)), (0,   0,   200)),
    ((int(82  * SCALE), int(126 * SCALE)), (0,   140, 255)),
    ((int(147 * SCALE), int(190 * SCALE)), (0,   200, 0  )),
    ((int(212 * SCALE), int(256 * SCALE)), (200, 0,   0  )),
    ((int(276 * SCALE), int(320 * SCALE)), (120, 0,  120)),
    ((int(350 * SCALE), int(385 * SCALE)), (30,  30,  30 )),
    ((int(405 * SCALE), int(449 * SCALE)), (220, 220, 220)),
]

toolbar = cv2.imread("Header/header.jpeg")
if toolbar is None:
    raise FileNotFoundError("Header/header.jpeg not found — check path")
toolbar = cv2.resize(toolbar, (CAM_WIDTH, HEADER_HEIGHT))

cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
detector = htm.handDetector(detectionCon=0.75, trackCon=0.75, maxHands=1)

# ── LAYERS ───────────────────────────────────────────────────
layers = LayerManager(CAM_HEIGHT, CAM_WIDTH)

# ── DRAWING STATE ────────────────────────────────────────────
activeTool      = "brush"
drawColor       = (0, 0, 200)
brushThickness  = 15
eraserThickness = 70
strokeStarted   = False
undoStack       = []
saveMsg         = 0
settingsOpen    = False
fps             = 30.0
handDetected    = False

liveStroke    = []    
currentStroke = []    

smoothX,smoothY       = None, None
lastRawX,lastRawY     = None, None
lastDrawX,lastDrawY   = None, None
ALPHA_S, ALPHA_F      = 0.18, 0.55

resizeCount    = 0
resizeStable   = 0
prevResizeDist = None

hRawTool  = None
hRawColor = None
hCount    = 0
hovTool   = None
hovColor  = None

clickCD   = 0
fistCount = 0

# ── AI STATE MACHINE ─────────────────────────────────────────
AI_IDLE      = 0
AI_ARMED     = 1   
AI_WAITING   = 2   
AI_ANIMATING = 3   
aiState      = AI_IDLE
pendingStroke= []
snap_anim    = SnapAnimation(duration_ms=220)
aiResult     = ""
aiResultTimer= 0

# ════════════════════════════════════════════════════════════
# PROFESSIONAL PHOTOSHOP-GRADE FLOOD FILL 
# ════════════════════════════════════════════════════════════
def perform_flood_fill(seed_x, seed_y, fill_color):
    """
    Executes a perfectly contained, leak-proof flood fill.
    Slides color smoothly under existing anti-aliased strokes without ruining transparency.
    """
    global aiResult, aiResultTimer, clickCD
    if seed_y <= HEADER_HEIGHT or seed_x < 0 or seed_x >= CAM_WIDTH or seed_y >= CAM_HEIGHT:
        return

    if np.array_equal(layers.layer0[seed_y, seed_x], fill_color):
        return

    # 1. Create a hard binary mask (ignoring anti-aliased gradients)
    gray = cv2.cvtColor(layers.layer0, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    h, w = binary_mask.shape
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # 2. Flood fill the BINARY mask space safely
    dry_run = binary_mask.copy()
    cv2.floodFill(dry_run, ff_mask, (seed_x, seed_y), 128)
    
    # 3. Leak check boundary validation
    if np.any(dry_run[HEADER_HEIGHT + 1, :] == 128) or np.any(dry_run[h - 1, :] == 128) or \
       np.any(dry_run[:, 0] == 128) or np.any(dry_run[:, w - 1] == 128):
        aiResult = "Fill Aborted: Shape is not closed!"
        aiResultTimer = 100
        clickCD = 25
        return

    # 4. Extract the exact filled region and dilate to slide UNDER the anti-aliased edge
    filled_region = (dry_run == 128).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    filled_region = cv2.dilate(filled_region, kernel, iterations=2)
    
    push_undo()
    
    # 5. FLAWLESS PRE-MULTIPLIED ALPHA COMPOSITING
    layer0_f = layers.layer0.astype(np.float32)
    fill_color_f = np.array(fill_color, dtype=np.float32)
    
    # Opacity is derived from the maximum color channel (0.0 to 1.0) so solid colors stay 100% solid
    opacity = np.max(layer0_f, axis=2, keepdims=True) / 255.0
    
    # Blend formula: Foreground (Stroke) + Background (Fill) * (1 - Opacity)
    blended = layer0_f + fill_color_f * (1.0 - opacity)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # 6. Apply the blended result ONLY inside the filled region. Leaves the rest of the canvas completely untouched!
    fill_mask_3d = filled_region[..., np.newaxis]
    layers.layer0 = np.where(fill_mask_3d == 1, blended, layers.layer0)
    
    clickCD = 35
    aiResult = "Area Filled Successfully"
    aiResultTimer = 60

# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════
def push_undo():
    undoStack.append(layers.copy_layer0())
    if len(undoStack) > 25:
        undoStack.pop(0)

def pop_undo():
    if undoStack:
        layers.restore_layer0(undoStack.pop())
        layers.cancel_snap()

def reset_mid_stroke():
    global strokeStarted,smoothX,smoothY,lastDrawX,lastDrawY,liveStroke
    strokeStarted=False; smoothX=smoothY=None
    lastDrawX=lastDrawY=None; liveStroke=[]

def check_and_end_stroke():
    global currentStroke, aiState
    if strokeStarted and len(liveStroke) > 2:
        currentStroke = list(liveStroke)
        if aiState == AI_ARMED:
            aiState = AI_WAITING
    reset_mid_stroke()

def smooth(rx,ry):
    global smoothX,smoothY,lastRawX,lastRawY
    if smoothX is None:
        smoothX,smoothY=rx,ry; lastRawX,lastRawY=rx,ry; return rx,ry
    d=math.hypot(rx-lastRawX,ry-lastRawY)
    a=ALPHA_S+(ALPHA_F-ALPHA_S)*min(d/40,1.0)
    smoothX+=a*(rx-smoothX); smoothY+=a*(ry-smoothY)
    lastRawX,lastRawY=rx,ry
    return smoothX,smoothY

def is_fist(f): return all(v==0 for v in f)

def fire_snap(stroke):
    global aiState,aiResult,aiResultTimer,pendingStroke,currentStroke
    push_undo()
    before=layers.composite()
    ok,name,conf=snap_from_stroke(layers,stroke,drawColor,brushThickness)
    if ok:
        after=layers.composite()
        snap_anim.start(before,after)
        pendingStroke=list(stroke)
        aiState=AI_ANIMATING
        aiResult=f"Snapped: {name}  |  Confidence {conf:.0%}"
    else:
        aiResult=name   
        aiState=AI_IDLE
    aiResultTimer=AI_RESULT_FRAMES
    currentStroke=[]

def clean_confirm_snap(pending_pts, thick):
    """
    Erases the original raw messy gesture stroke completely 
    from layer0 before applying the clean AI-snapped geometry.
    """
    if len(pending_pts) >= 2:
        mask = np.zeros((layers.h, layers.w), dtype=np.uint8)
        for i in range(1, len(pending_pts)):
            cv2.line(mask, pending_pts[i-1], pending_pts[i], 255, thickness=thick + 16, lineType=cv2.LINE_AA)
        layers.layer0[mask > 0] = (0, 0, 0)
        
    layers.confirm_snap(pending_pts, thick)

# ════════════════════════════════════════════════════════════
# HOVER & RESIZE
# ════════════════════════════════════════════════════════════
def update_hover(x,y):
    global hRawTool,hRawColor,hCount,hovTool,hovColor
    rt=rc=None
    if y<HEADER_HEIGHT:
        for tool,(xs,xe) in TOOL_ZONES.items():
            if xs<x<xe:
                if tool=="colors":
                    for zone in COLOR_ZONES:
                        (cx1,cx2),c=zone
                        if cx1<x<cx2: rc=zone
                else: rt=tool
                break
    if rt==hRawTool and rc==hRawColor: hCount+=1
    else: hRawTool=rt; hRawColor=rc; hCount=0
    if hCount>=HOVER_DEBOUNCE: hovTool=rt; hovColor=rc

def handle_resize(lm,fingers,frame):
    global resizeCount,resizeStable,prevResizeDist,brushThickness,eraserThickness
    tip=(fingers[0]==1 and fingers[1]==1 and
         fingers[2]==0 and fingers[3]==0 and fingers[4]==0)
    if not tip:
        resizeCount=resizeStable=0; prevResizeDist=None; return False
    resizeCount+=1
    if resizeCount<RESIZE_DEBOUNCE: return True
    x1,y1=lm[4][1],lm[4][2]; x2,y2=lm[8][1],lm[8][2]
    dist=math.hypot(x2-x1,y2-y1)
    if prevResizeDist is not None and abs(dist-prevResizeDist)<4: resizeStable+=1
    else: resizeStable=0
    prevResizeDist=dist
    if activeTool=="eraser":
        eraserThickness=int(np.interp(dist,[20,200],[MIN_ERASER,MAX_ERASER])); sz=eraserThickness
    else:
        brushThickness=int(np.interp(dist,[20,200],[MIN_BRUSH,MAX_BRUSH])); sz=brushThickness
    cx,cy=(x1+x2)//2,(y1+y2)//2
    cv2.line(frame,(x1,y1),(x2,y2),HIGHLIGHT_COLOR,2)
    cv2.circle(frame,(cx,cy),max(sz//2,2),drawColor,cv2.FILLED)
    cv2.circle(frame,(cx,cy),max(sz//2,2),(200,200,200),2)
    label="LOCKED" if resizeStable>RESIZE_LOCK else f"{sz}px"
    cv2.putText(frame,label,(cx-25,cy-sz//2-14),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    return True

# ════════════════════════════════════════════════════════════
# TOOLBAR
# ════════════════════════════════════════════════════════════
def handle_toolbar_click(x,y):
    global activeTool,drawColor,saveMsg,settingsOpen,aiState,currentStroke,clickCD
    if clickCD>0: return
    for tool,(xs,xe) in TOOL_ZONES.items():
        if xs<x<xe:
            clickCD=CLICK_COOLDOWN
            if tool=="colors":
                for (cx1,cx2),color in COLOR_ZONES:
                    if cx1<x<cx2:
                        drawColor=color
                        if activeTool!="eraser": activeTool="brush"
                        return
                return
            elif tool=="undo": pop_undo()
            elif tool=="clear":
                push_undo(); layers.clear()
                currentStroke=[]; aiState=AI_IDLE
            elif tool=="save":
                cv2.imwrite("saved_drawing.png",layers.composite()); saveMsg=90
            elif tool=="ai":
                if aiState==AI_IDLE:
                    aiState=AI_ARMED
                    if len(currentStroke)>3:
                        aiState=AI_WAITING
                elif aiState in (AI_ARMED,AI_WAITING):
                    aiState=AI_IDLE; layers.cancel_snap()
            elif tool=="settings": settingsOpen=not settingsOpen
            else:
                activeTool=tool; aiState=AI_IDLE
                layers.cancel_snap(); settingsOpen=False
            return

# ── UI overlay utils ─────────────────────────────────────────
AI_COLORS = {AI_ARMED:(0,170,70), AI_WAITING:(0,200,255), AI_ANIMATING:(200,150,0)}

def draw_toolbar(frame,in_panel=False,px=0,py=0):
    frame[0:HEADER_HEIGHT,0:CAM_WIDTH]=toolbar
    if activeTool in TOOL_ZONES:
        xs,xe=TOOL_ZONES[activeTool]
        cv2.rectangle(frame,(xs+2,4),(xe-2,HEADER_HEIGHT-4),HIGHLIGHT_COLOR,3,cv2.LINE_AA)
    if aiState!=AI_IDLE:
        col=AI_COLORS.get(aiState,(200,200,200))
        xs,xe=TOOL_ZONES["ai"]
        cv2.rectangle(frame,(xs+2,4),(xe-2,HEADER_HEIGHT-4),col,3,cv2.LINE_AA)
        labels={AI_ARMED:"DRAW",AI_WAITING:"FIST",AI_ANIMATING:"..."}
        cv2.putText(frame,labels.get(aiState,""),(xs+6,HEADER_HEIGHT-6),cv2.FONT_HERSHEY_SIMPLEX,0.36,col,1)
    if hovTool and hovTool!=activeTool and not(hovTool=="ai" and aiState!=AI_IDLE) and hovTool!="colors":
        xs,xe=TOOL_ZONES[hovTool]
        cv2.rectangle(frame,(xs+2,4),(xe-2,HEADER_HEIGHT-4),HOVER_COLOR,2,cv2.LINE_AA)
    for (cx1,cx2),color in COLOR_ZONES:
        if color==drawColor:
            cv2.rectangle(frame,(cx1,COLOR_Y_START),(cx2,COLOR_Y_END),HIGHLIGHT_COLOR,2,cv2.LINE_AA)
    if hovColor:
        (hx1,hx2),hc=hovColor
        if hc!=drawColor:
            cv2.rectangle(frame,(hx1,COLOR_Y_START),(hx2,COLOR_Y_END),HOVER_COLOR,2,cv2.LINE_AA)
    if in_panel:
        cv2.rectangle(frame,(px-18,py-18),(px+18,py+18),PANEL_CURSOR_COLOR,2,cv2.LINE_AA)

def draw_ai_hint(frame):
    hints={
        AI_ARMED:   "AI READY — Draw your shape, then lift finger",
        AI_WAITING: "Shape captured — Make a FIST to snap  |  Tap AI to cancel",
        AI_ANIMATING: "Snapping...",
    }
    text=hints.get(aiState,""); col=AI_COLORS.get(aiState,(200,200,200))
    if not text: return
    px,py=CAM_WIDTH//2-270,HEADER_HEIGHT+8
    ov=frame.copy()
    cv2.rectangle(ov,(px,py),(px+540,py+34),(15,15,15),cv2.FILLED)
    cv2.addWeighted(ov,0.85,frame,0.15,0,frame)
    cv2.rectangle(frame,(px,py),(px+540,py+34),col,1)
    cv2.putText(frame,text,(px+10,py+22),cv2.FONT_HERSHEY_SIMPLEX,0.42,col,1)
    if aiState==AI_WAITING and fistCount>0:
        pct=fistCount/FIST_CONFIRM
        bx1,by1=px,py+34; bx2=px+int(540*pct); by2=py+38
        cv2.rectangle(frame,(bx1,by1),(px+540,by2),(35,35,35),cv2.FILLED)
        cv2.rectangle(frame,(bx1,by1),(bx2,by2),col,cv2.FILLED)

def draw_settings_panel(frame):
    sx,_=TOOL_ZONES["settings"]
    px,py=sx,HEADER_HEIGHT+4; pw=CAM_WIDTH-sx; ph=108
    ov=frame.copy()
    cv2.rectangle(ov,(px,py),(px+pw,py+ph),(16,16,16),cv2.FILLED)
    cv2.addWeighted(ov,0.90,frame,0.10,0,frame)
    cv2.rectangle(frame,(px,py),(px+pw,py+ph),HOVER_COLOR,1)
    sz=eraserThickness if activeTool=="eraser" else brushThickness
    for i,ln in enumerate([f"Tool  : {activeTool.capitalize()}",
                            f"Size  : {sz}px",
                            f"FPS   : {fps:.0f}",
                            f"Hand  : {'Yes' if handDetected else 'No'}"]):
        cv2.putText(frame,ln,(px+10,py+22+i*22),cv2.FONT_HERSHEY_SIMPLEX,0.46,(170,170,170),1)
    cv2.circle(frame,(px+pw-14,py+14),8,drawColor,cv2.FILLED)
    cv2.circle(frame,(px+pw-14,py+14),8,(90,90,90),1)

def draw_result(frame,text):
    ov=frame.copy()
    bx,by=40,CAM_HEIGHT-108
    cv2.rectangle(ov,(bx,by),(CAM_WIDTH-40,CAM_HEIGHT-14),(14,14,14),cv2.FILLED)
    cv2.addWeighted(ov,0.84,frame,0.16,0,frame)
    parts=text.split("|")
    cv2.putText(frame,parts[0].strip(),(bx+14,by+38),cv2.FONT_HERSHEY_SIMPLEX,0.95,(0,210,85),2,cv2.LINE_AA)
    if len(parts)>1:
        cv2.putText(frame,parts[1].strip(),(bx+14,by+66),cv2.FONT_HERSHEY_SIMPLEX,0.55,(120,190,120),1,cv2.LINE_AA)

# ════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════
prevTime=time.time()

while True:
    success,img=cap.read()
    if not success: break
    img=cv2.flip(img,1)

    now=time.time()
    fps=0.9*fps+0.1*(1/(now-prevTime+1e-9))
    prevTime=now
    if clickCD>0: clickCD-=1

    img=detector.findHands(img,draw=False)
    lmList,_=detector.findPosition(img,draw=False)
    handDetected=len(lmList)>=21

    comp=layers.composite()
    if snap_anim.active:
        comp=snap_anim.update(comp)
        if snap_anim.done and aiState==AI_ANIMATING:
            clean_confirm_snap(pendingStroke, brushThickness)
            aiState=AI_IDLE

    gray=cv2.cvtColor(comp,cv2.COLOR_BGR2GRAY)
    _,inv=cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)
    inv=cv2.cvtColor(inv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,inv)
    img=cv2.bitwise_or(img,comp)

    in_panel=False; panX=panY=0

    if not handDetected:
        check_and_end_stroke()
        resizeCount=0; fistCount=0
        hovTool=hovColor=None; hCount=0
    else:
        ix,iy=lmList[8][1],lmList[8][2]   
        mx,my=lmList[12][1],lmList[12][2] 
        fingers=detector.fingersUp()

        update_hover(ix,iy)
        if iy<HEADER_HEIGHT: in_panel=True; panX,panY=ix,iy

        # ── P1: resize ───────────────────────────────────────
        if handle_resize(lmList,fingers,img):
            check_and_end_stroke()
            fistCount=0

        # ── P2.5: three-finger fill (Index + Middle + Ring) ──
        elif fingers[1] and fingers[2] and fingers[3] and not fingers[4]:
            check_and_end_stroke()
            fistCount = 0
            if iy > HEADER_HEIGHT:
                cv2.drawMarker(img, (ix, iy), drawColor, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
                cv2.circle(img, (ix, iy), 5, (235, 235, 235), 1, cv2.LINE_AA)
                if clickCD == 0:
                    perform_flood_fill(ix, iy, drawColor)

        # ── P2: two-finger selection ─────────────────────────
        elif fingers[1] and fingers[2] and not fingers[3]:
            check_and_end_stroke()
            fistCount=0
            col=PANEL_CURSOR_COLOR if iy<HEADER_HEIGHT else HIGHLIGHT_COLOR
            cv2.rectangle(img,(min(ix,mx)-10,min(iy,my)-10),(max(ix,mx)+10,max(iy,my)+10),col,2)
            if iy<HEADER_HEIGHT:
                handle_toolbar_click(ix,iy)

        # ── P3: fist ─────────────────────────────────────────
        elif is_fist(fingers):
            fistCount+=1
            draw_ai_hint(img)   
            if fistCount==FIST_CONFIRM and aiState==AI_WAITING:
                if len(currentStroke)>3:
                    fire_snap(currentStroke)
                else:
                    aiResult="No stroke to snap — draw first"
                    aiResultTimer=120; aiState=AI_IDLE
            check_and_end_stroke()

        # ── P4: draw (index finger only) ─────────────────────
        elif fingers[1] and not fingers[2]:
            fistCount=0
            if iy>HEADER_HEIGHT:
                sx,sy=smooth(ix,iy); sx,sy=int(sx),int(sy)
                if not strokeStarted:
                    push_undo(); strokeStarted=True
                    lastDrawX,lastDrawY=sx,sy; liveStroke=[(sx,sy)]
                th=eraserThickness if activeTool=="eraser" else brushThickness
                cl=(0,0,0)         if activeTool=="eraser" else drawColor
                if lastDrawX is not None:
                    if activeTool=="eraser":
                        layers.erase_stroke((lastDrawX,lastDrawY),(sx,sy),th)
                    else:
                        layers.draw_stroke((lastDrawX,lastDrawY),(sx,sy),cl,th)
                liveStroke.append((sx,sy))
                lastDrawX,lastDrawY=sx,sy
            cv2.circle(img,(ix,iy),10,drawColor,cv2.FILLED)

        # ── stroke ended vectors ─────────────────────────────
        else:
            fistCount=0
            check_and_end_stroke()

    # ── UI overlay ───────────────────────────────────────────
    draw_toolbar(img,in_panel,panX,panY)
    if settingsOpen: draw_settings_panel(img)
    if aiState in (AI_ARMED,AI_WAITING,AI_ANIMATING): draw_ai_hint(img)
    if aiResult and aiResultTimer>0:
        draw_result(img,aiResult); aiResultTimer-=1
    if saveMsg>0:
        cv2.putText(img,"SAVED!",(int(CAM_WIDTH/2)-60,int(CAM_HEIGHT/2)),cv2.FONT_HERSHEY_SIMPLEX,2,(0,210,85),4)
        saveMsg-=1

    cv2.imshow("GestureCanvas",img)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'): break
    elif key==ord('z'): pop_undo()
    elif key==ord('s'):
        cv2.imwrite("saved_drawing.png",layers.composite()); saveMsg=90

cap.release()
cv2.destroyAllWindows()