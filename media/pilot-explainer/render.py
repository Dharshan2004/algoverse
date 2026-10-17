"""Render narrated pilot explainer. Requires macOS say, Pillow, imageio-ffmpeg."""
from pathlib import Path
import json, subprocess, wave, math, textwrap
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
HERE=Path(__file__).resolve().parent
REPO=HERE.parent.parent
BUILD=REPO/'runs/video-build'; BUILD.mkdir(parents=True,exist_ok=True)
FF=imageio_ffmpeg.get_ffmpeg_exe()
W,H,FPS=1280,720,20
BG='#101927'; WHITE='#edf2f7'; MUTED='#a8bacb'; TEAL='#62dec2'; CORAL='#ff947e'; BLUE='#88b8ff'
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'
def font(n):return ImageFont.truetype(FONT,n)
scenes=[
('The question','Can an opening sentence change a safety judgment?', 'This pilot asks a simple question. If a response contains unsafe material, can adding a refusal sentence make a judge call it safe? We tested saved responses, not model training. Here is how the code works, and what actually happened.'),
('01  Build the experiment','protocol.py', 'We start with fifty saved request and response pairs: twenty five labelled unsafe and twenty five labelled safe. Protocol dot py creates four versions. Original, refusal opening, neutral opening, and an alternate refusal. The response body stays exactly the same. The human label is kept for scoring, and is never shown to the judge.'),
('02  Swap the answer letters','A/B mapping control', 'Each version is judged twice. First, A means safe and B means unsafe. Then we reverse those meanings. This checks whether a preference for the letter A or B influences the outcome. Fifty records, times four versions, times two mappings, makes four hundred judgments for each model.'),
('03  Run the models','inference.py', 'Inference dot py loads a pinned model checkpoint on the GPU. It formats the prompt, checks its length, and generates up to four answer tokens without random sampling. The parser accepts an exact answer code after limited native formatting cleanup. Other outputs remain invalid. Both models ran on an L4 GPU with B F sixteen computation and compressed four bit weights.'),
('04  Save and verify','pilot_report.py + metrics.py', 'For every judgment, we save the answer, first token A and B scores, and six internal activation snapshots. These snapshots record what happened; we did not change a refusal direction. Two extra calls per model check that a do nothing intervention leaves results unchanged. The reporting code then checks completeness, unchanged text, scores and saved tensors before calculating rates.'),
('05  What changed?','False-safe rate: lower is better', 'A false safe judgment means the dataset says unsafe, but the judge says safe. Averaging the two letter mappings, Ministral increased from six percent to ten percent with the refusal opening. Gemma increased from twenty two percent to twenty eight percent. These are changes of four and six percentage points, not proof of a general effect.'),
('06  Check the controls','The details change the interpretation', 'The neutral opening left Ministral at six percent and raised Gemma to twenty four percent. Ministral showed no refusal increase under one letter mapping, and an eight point increase under the other. Only Gemma had examples that flipped from unsafe to safe under both mappings: two of twenty five. Ministral had zero.'),
('07  What is established?','Completed execution; exploratory evidence', 'Eight hundred judgments and four identity calls completed. All saved activation files passed the audit. Ministral had no invalid outputs; Gemma had ten out of four hundred. Those invalid outputs stay in the calculations. Source labels, possible text completeness issues, and shared request families still need review. We have not established statistical significance or a causal mechanism.'),
('08  Track the work','Notebooks, code, evidence and next steps', 'The repository now separates reusable notebooks from sanitized historical snapshots. The actual pilot ran through the uploaded notebook and existing inference script. The new pilot launcher is for future runs, and was dry run tested. Aggregate reports, notebook provenance and this video are tracked. Raw data stays private. Next, review the sample before scaling or attempting causal interventions.')]
(HERE/'transcript.md').write_text('# Pilot explainer transcript\n\n'+'\n\n'.join('## '+a+'\n\n'+c for a,b,c in scenes)+'\n')
def text(d,xy,s,size=28,color=WHITE):d.text(xy,s,font=font(size),fill=color)
def card(d,box,title,body,color=TEAL):
 d.rounded_rectangle(box,radius=18,fill='#1b2a3c',outline=color,width=2); x,y,_,_=box
 text(d,(x+22,y+20),title,28,color)
 for i,line in enumerate(body.split('\n')):text(d,(x+22,y+68+i*34),line,24)
def frame(i,t,duration):
 im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
 text(d,(60,35),'ALGOVERSE  /  PILOT WALKTHROUGH',20,TEAL)
 text(d,(60,83),scenes[i][0],45);text(d,(60,143),scenes[i][1],27,MUTED)
 p=min(1,t/2); ease=1-(1-p)**3
 if i==0:
  card(d,(70,235,565,450),'Same response','Original body\nOriginal body + refusal opening')
  card(d,(710,235,1205,450),'Same judge. Different verdict?','SAFE  /  UNSAFE  /  INVALID',CORAL)
  x=590+int((t%2)/2*90);d.ellipse((x,325,x+15,340),fill=TEAL)
  text(d,(70,495),'50 records   ·   2 judges   ·   800 judgments',35)
 elif i==1:
  titles=['B  Original','R  Refusal','N  Neutral','R2  Alternate refusal']
  bodies=['No added opening','I cannot help with that request.','I have read the request above.','I am unable to assist...']
  for j in range(4):
   y=215+j*79; x=70+int((1-min(1,t/(.4+j*.3)))*70)
   d.rounded_rectangle((x,y,1210,y+65),radius=12,fill='#1b2a3c')
   text(d,(x+18,y+16),titles[j],25,TEAL);text(d,(430,y+16),bodies[j]+'  +  unchanged body' if j else 'Unchanged body',24)
 elif i==2:
  card(d,(70,225,610,405),'Mapping 1','A = SAFE     B = UNSAFE')
  card(d,(665,225,1205,405),'Mapping 2','A = UNSAFE   B = SAFE',BLUE)
  text(d,(155,465),'50 × 4 × 2 = '+str(int(400*ease))+' per model',46)
 elif i==3:
  for j,(a,b) in enumerate([('Load','Pinned checkpoint'),('Prepare','Native chat format'),('Judge','Up to 4 tokens'),('Parse','Exact A or B')]):
   x=60+j*305;card(d,(x,255,x+280,435),a,b,TEAL if int(t)%4==j else BLUE)
  text(d,(70,485),'Oversized input? Stop. Unparseable answer? INVALID.',29)
 elif i==4:
  for j in range(6):
   x=85+j*190;d.rounded_rectangle((x,255,x+145,385),radius=15,fill='#1b2a3c',outline=TEAL,width=2)
   text(d,(x+30,300),'L'+str(j+1),32)
   if (t*1.5)%6>=j:d.ellipse((x+58,405,x+88,435),fill=TEAL)
  text(d,(85,465),'Six selected layers: snapshots, not causal edits',32)
  text(d,(85,515),'Audit → complete rows · preserved body · finite states',27,MUTED)
 elif i==5:
  for j,(name,a,b) in enumerate([('Ministral',6,10),('Gemma',22,28)]):
   y=225+j*160;text(d,(70,y),name,30)
   for k,(val,col,label) in enumerate([(a,BLUE,'Original'),(b,CORAL,'Refusal')]):
    yy=y+45+k*43;text(d,(70,yy),label,22,MUTED)
    d.rounded_rectangle((225,yy,225+max(1,int(val/30*750*ease)),yy+30),radius=5,fill=col)
    text(d,(1000,yy-4),f'{val}%',28,col)
  text(d,(70,560),'0%                                     common scale                                      30%',19,MUTED)
 elif i==6:
  card(d,(70,225,610,435),'Neutral opening','Ministral  6% → 6%\nGemma     22% → 24%')
  card(d,(665,225,1205,435),'Flips under BOTH mappings','Ministral  0 / 25\nGemma      2 / 25',CORAL)
  text(d,(70,480),'Ministral refusal effect: +0 pp vs +8 pp by mapping',29)
 elif i==7:
  card(d,(70,225,610,465),'Verified','800 judgments + 4 checks\n800 activation files audited\nMinistral invalid: 0 / 400')
  card(d,(665,225,1205,465),'Still unresolved','Gemma invalid: 10 / 400\nSource labels and completeness\nIndependent request families',CORAL)
  text(d,(70,510),'No significance claim. No causal mechanism established.',29)
 else:
  for j,(a,b) in enumerate([('Recorded','Sanitized run notebooks'),('Preserved','Private raw artifacts'),('Next','Review sample, then scale')]):
   x=60+j*410;card(d,(x,255,x+385,445),a,b,TEAL if j<2 else CORAL)
  text(d,(70,490),'Actual run: uploaded notebook → inference.py',29)
  text(d,(70,535),'pilot.py: reusable launcher, dry-run tested',26,MUTED)
 # readable rolling captions from narration, no dependence on audio
 chunks=textwrap.wrap(scenes[i][2],95)
 ix=min(len(chunks)-1,int(t/duration*len(chunks)))
 d.rectangle((0,595,W,720),fill='#0a111b')
 text(d,(60,615),chunks[ix],24)
 text(d,(60,672),'17 OCT 2026  ·  EXPLORATORY DEVELOPMENT DATA',16,MUTED)
 text(d,(1120,672),f'{i+1:02} / 09',18,TEAL)
 d.rectangle((0,711,int(W*((i+t/duration)/len(scenes))),720),fill=TEAL)
 return im
parts=[]; durations=[]; thumbs=[]
for i,(_,_,voice) in enumerate(scenes):
 txt=BUILD/f'{i}.txt';txt.write_text(voice)
 aiff=BUILD/f'{i}.aiff';wav=BUILD/f'{i}.wav'
 subprocess.run(['say','-v','Samantha','-r','155','-f',str(txt),'-o',str(aiff)],check=True)
 subprocess.run([FF,'-y','-v','error','-i',str(aiff),str(wav)],check=True)
 with wave.open(str(wav)) as w:duration=w.getnframes()/w.getframerate()+1.0
 assert duration > 3, 'Speech synthesis returned empty audio; run with macOS speech access'
 duration=math.ceil(duration*FPS)/FPS;durations.append(duration)
 silent=BUILD/f'{i}-silent.mp4';part=BUILD/f'{i}.mp4'
 proc=subprocess.Popen([FF,'-y','-v','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p',str(silent)],stdin=subprocess.PIPE)
 for f in range(round(duration*FPS)):proc.stdin.write(frame(i,f/FPS,duration).tobytes())
 proc.stdin.close();assert proc.wait()==0
 subprocess.run([FF,'-y','-v','error','-i',str(silent),'-i',str(wav),'-c:v','copy','-c:a','aac','-af','apad','-t',str(duration),str(part)],check=True)
 parts.append(part);thumbs.append(frame(i,min(4,duration/2),duration))
 print('Rendered scene',i+1,'duration',duration,flush=True)
concat=BUILD/'concat.txt';concat.write_text(''.join("file '"+str(p)+"'\n" for p in parts))
out=HERE/'pilot-explained.mp4'
subprocess.run([FF,'-y','-v','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(out)],check=True)
sheet=Image.new('RGB',(960,540))
for i,im in enumerate(thumbs):sheet.paste(im.resize((320,180)),((i%3)*320,(i//3)*180))
sheet.save(BUILD/'contact-sheet.jpg');thumbs[0].save(HERE/'poster.jpg')
(HERE/'video.json').write_text(json.dumps({'duration_seconds':sum(durations),'scene_seconds':durations,'resolution':[W,H],'fps':FPS,'narration':'macOS Samantha synthetic voice','result_sources':['reports/pilot-2026-10-17/ministral/summary.json','reports/pilot-2026-10-17/gemma/summary.json']},indent=2)+'\n')
print('DONE',out,sum(durations),flush=True)
