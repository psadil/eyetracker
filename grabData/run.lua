#!/usr/bin/env torch
------------------------------------------------------------
--
-- CNN face detector, based on convolutional network nets
--
-- original: Clement Farabet
-- E. Culurciello, A. Dundar, A. Canziani
-- Tue Mar 11 10:52:58 EDT 2014
--
------------------------------------------------------------

require 'pl'
require 'qt'
require 'qtwidget'
require 'qtuiloader'
require 'camera'
require 'image'
require 'nnx'
require 'sys'
require 'torchx'
require 'dp'
print '==> processing options'

opt = lapp[[
   -c, --camidx   (default 0)             camera index: /dev/videoIDX
   -t, --threads  (default 8)             number of threads
       --HD       (default true)          high resolution camera
]]


torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.threads)


-- camera
local GUI
if opt.HD then
   camera = image.Camera{idx=opt.camidx, width=96, height=96} -- note, camera going to 120x160
   GUI = 'HDg.ui'
else
   camera = image.Camera(opt.camidx)
   GUI = 'g.ui'
end

-- setup GUI (external UI file)
if not win or not widget then
   widget = qtuiloader.load(GUI)
   win = qt.QtLuaPainter(widget.frame)
end


-- profiler
p = xlua.Profiler()


--function process()
   -- (1) grab frame
--   frame = camera:forward()
--end

img = image.load('jelly-eye.jpg')
img = image.scale(img, 96,96)

mu = torch.ones(2) * 96


--Data dir-- 
dataPath = 'data/'
gathered = paths.indexdir(dataPath)
idx = gathered:size()+1
--idx = 1
nToDo = 1200	

--labels = torch.zeros(nToDo,2)
----labels = torch.zeros(gathered:size()+nToDo,2)
labels_old = torch.load('data/labels.dat')
labels = torch.cat(labels_old, torch.zeros(nToDo,2),1)


--for i=1,25 do
--	print(gathered:filename(i))
--end

-- display function
function display()
	win:gbegin()
   	win:showpage()
   	-- (1) display input image + pyramid
   	--image.display{image=frame, win=win}
   	--image.display{image=img, win=win}

   	local xy = torch.ceil(torch.cmul(torch.rand(2),mu))
   	print(xy)

   	win:showpage()
   	win:arc(xy[1],xy[2],10,0,360)
   	if torch.all(torch.le(torch.rand(1),.8)) then
   		-- draw normal green dot to track
		win:setcolor("green"); 
		labels[{{idx},{}}] = xy:clone()
--   	else
   		-- say BLINK
		--win:setcolor("red");
--		win:moveto(200,200)
--		win:setcolor("red")
--		win:setfont(qt.QFont{serif=true,size=50})
--		win:show("BLINK")
--		labels[{{idx},{}}] = torch.Tensor({{100},{101}})
   	end
   	win:fill(true)

	print(idx)

	sys.sleep(.5)
	frame = camera:forward()
	image.savePNG('data/eye_'..idx..'.png', frame)
	

	torch.save('data/labels.dat',labels)
	win:gend()

	idx = idx+1
end



-- setup gui
timer = qt.QTimer()
timer.interval = 1
timer.singleShot = true

--wtTime = qt.QTimer()
--wtTime.singleShot = true


qt.connect(timer,
           'timeout()',
           function()
              p:start('full loop','fps')
              p:start('prediction','fps')
              --process()
              p:lap('prediction')
              p:start('display','fps')
              display()
              p:lap('display')
              timer:start()
              p:lap('full loop')
              p:printAll()

           end)
widget.windowTitle = 'e-Lab Face Detector'
widget:show()
timer:start()


