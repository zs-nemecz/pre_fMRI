#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on december 04, 2024, at 10:30
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'learning'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1440, 900]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Asus\\Documents\\pretest_fmri\\pre_fMRI\\experimetal_task_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('instruction1_key') is None:
        # initialise instruction1_key
        instruction1_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction1_key',
        )
    if deviceManager.getDevice('instruction2_key') is None:
        # initialise instruction2_key
        instruction2_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction2_key',
        )
    if deviceManager.getDevice('instruction3_key') is None:
        # initialise instruction3_key
        instruction3_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction3_key',
        )
    if deviceManager.getDevice('instruction4_key') is None:
        # initialise instruction4_key
        instruction4_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instruction4_key',
        )
    if deviceManager.getDevice('end_guess') is None:
        # initialise end_guess
        end_guess = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_guess',
        )
    if deviceManager.getDevice('guess_reached') is None:
        # initialise guess_reached
        guess_reached = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='guess_reached',
        )
    if deviceManager.getDevice('end_encoding') is None:
        # initialise end_encoding
        end_encoding = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_encoding',
        )
    if deviceManager.getDevice('start_key') is None:
        # initialise start_key
        start_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_key',
        )
    if deviceManager.getDevice('end_part1_key') is None:
        # initialise end_part1_key
        end_part1_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_part1_key',
        )
    if deviceManager.getDevice('recall_instructions_key') is None:
        # initialise recall_instructions_key
        recall_instructions_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='recall_instructions_key',
        )
    if deviceManager.getDevice('end_cued_recall') is None:
        # initialise end_cued_recall
        end_cued_recall = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_cued_recall',
        )
    if deviceManager.getDevice('recall_reached') is None:
        # initialise recall_reached
        recall_reached = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='recall_reached',
        )
    if deviceManager.getDevice('recall_selection') is None:
        # initialise recall_selection
        recall_selection = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='recall_selection',
        )
    if deviceManager.getDevice('end_part2_key') is None:
        # initialise end_part2_key
        end_part2_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_part2_key',
        )
    if deviceManager.getDevice('sm_instructions_key') is None:
        # initialise sm_instructions_key
        sm_instructions_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='sm_instructions_key',
        )
    if deviceManager.getDevice('living_nonliving') is None:
        # initialise living_nonliving
        living_nonliving = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='living_nonliving',
        )
    if deviceManager.getDevice('end_mapping') is None:
        # initialise end_mapping
        end_mapping = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_mapping',
        )
    if deviceManager.getDevice('break_resp') is None:
        # initialise break_resp
        break_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='break_resp',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='Welcome to the Guessing Game Experiment!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from randomize_guesses
    import random as rnd
    
    practice_list = 2 * ["Guess", "Read"]
    rnd.shuffle(practice_list)
    guess_list = 39 * ["Guess", "Read"]
    print(guess_list)
    rnd.shuffle(guess_list)
    print(guess_list)
    
    
    # --- Initialize components for Routine "instructions1" ---
    instructions1_text = visual.TextStim(win=win, name='instructions1_text',
        text='Part 1\n\nIn the first part of the experiment, you will read word pairs on the screen. \n\nFor example:\n\nComputer - Keyboard\n\nYour task is to try to memorize as many of the word pairs as possible.\n\nPress SPACE to continue.\n\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instruction1_key = keyboard.Keyboard(deviceName='instruction1_key')
    
    # --- Initialize components for Routine "instructions2" ---
    instructions2_text = visual.TextStim(win=win, name='instructions2_text',
        text='Sometimes, before you read the full word pair, you will see a single word on the screen and you will have to guess what the other word is.\n\nFor example:\n\nComputer - ?\n\nPress SPACE to continue.\n\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instruction2_key = keyboard.Keyboard(deviceName='instruction2_key')
    
    # --- Initialize components for Routine "instructions3" ---
    instructions3_text = visual.TextStim(win=win, name='instructions3_text',
        text='Do not say the guess out loud but try to generate as many guesses silently as possible. You will have 5 seconds to make a guess, then you will have to indicate whether you could generate one or more guesses with the keys F,G,H.\n\nF) Made no guess        G) Made 1 guess        H) Made more than 1 guess\n\nPress SPACE to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instruction3_key = keyboard.Keyboard(deviceName='instruction3_key')
    
    # --- Initialize components for Routine "instructions4" ---
    instructions4_text = visual.TextStim(win=win, name='instructions4_text',
        text='Summary: \nYour task is to learn as many of the word pairs presented on the screen as possible.\n\nSometimes, before you see the word pair, you will see one word and you will have to guess what the pair of the word is. Press F, G,H to indicate whether you came up with a guess.\n\nBefore the experiment, you will practice the task.\n\nPress SPACE to start the practice.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instruction4_key = keyboard.Keyboard(deviceName='instruction4_key')
    
    # --- Initialize components for Routine "iti_learning_practice" ---
    ISI_4 = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI_4')
    
    # --- Initialize components for Routine "guess" ---
    dash_stim_guessing = visual.TextStim(win=win, name='dash_stim_guessing',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_guessing = visual.TextStim(win=win, name='cue_stim_guessing',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    guess_stim = visual.TextStim(win=win, name='guess_stim',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    end_guess = keyboard.Keyboard(deviceName='end_guess')
    
    # --- Initialize components for Routine "guess_response" ---
    dash_stim_guessing_resp = visual.TextStim(win=win, name='dash_stim_guessing_resp',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_guessing_resp = visual.TextStim(win=win, name='cue_stim_guessing_resp',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    guess_stim_resp = visual.TextStim(win=win, name='guess_stim_resp',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    guess_question = visual.TextStim(win=win, name='guess_question',
        text='How many guesses did you make?\n\nF) 0        G) 1        H) More',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    guess_reached = keyboard.Keyboard(deviceName='guess_reached')
    
    # --- Initialize components for Routine "feedback_guess" ---
    no_response_feedback = visual.TextStim(win=win, name='no_response_feedback',
        text='You did not press F or J.\n\nRemember to press key J to indicate that you came up with a guess.\nPress the F key if you could not come up with a guess.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "encode" ---
    dash_stim_learning = visual.TextStim(win=win, name='dash_stim_learning',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_learning = visual.TextStim(win=win, name='cue_stim_learning',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    target_stim = visual.TextStim(win=win, name='target_stim',
        text='',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    end_encoding = keyboard.Keyboard(deviceName='end_encoding')
    
    # --- Initialize components for Routine "start_task" ---
    start_task_text = visual.TextStim(win=win, name='start_task_text',
        text='Now comes the task. \n\nPress SPACE when you are ready to start.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    start_key = keyboard.Keyboard(deviceName='start_key')
    
    # --- Initialize components for Routine "iti_learning" ---
    ISI = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI')
    
    # --- Initialize components for Routine "guess" ---
    dash_stim_guessing = visual.TextStim(win=win, name='dash_stim_guessing',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_guessing = visual.TextStim(win=win, name='cue_stim_guessing',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    guess_stim = visual.TextStim(win=win, name='guess_stim',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    end_guess = keyboard.Keyboard(deviceName='end_guess')
    
    # --- Initialize components for Routine "guess_response" ---
    dash_stim_guessing_resp = visual.TextStim(win=win, name='dash_stim_guessing_resp',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_guessing_resp = visual.TextStim(win=win, name='cue_stim_guessing_resp',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    guess_stim_resp = visual.TextStim(win=win, name='guess_stim_resp',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    guess_question = visual.TextStim(win=win, name='guess_question',
        text='How many guesses did you make?\n\nF) 0        G) 1        H) More',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    guess_reached = keyboard.Keyboard(deviceName='guess_reached')
    
    # --- Initialize components for Routine "encode" ---
    dash_stim_learning = visual.TextStim(win=win, name='dash_stim_learning',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_learning = visual.TextStim(win=win, name='cue_stim_learning',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    target_stim = visual.TextStim(win=win, name='target_stim',
        text='',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    end_encoding = keyboard.Keyboard(deviceName='end_encoding')
    
    # --- Initialize components for Routine "end_part1" ---
    end_part1_key = keyboard.Keyboard(deviceName='end_part1_key')
    end_part1_text = visual.TextStim(win=win, name='end_part1_text',
        text='You just finished Part 1 of the experiment!\n\nTake a 5 minute break before continuing.\n\nPress SPACE to continue ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "instructions_recall" ---
    recall_instructions_text = visual.TextStim(win=win, name='recall_instructions_text',
        text="In the next part of the experiment, you will have to recall the pair of each word you see on the screen.\n\nTry to recall the word (5s) and then indicate whether you remember the word with the keys F (Can't recall) and G (Can recall).\n\nIf you could recall the word, we will ask you what the last letter of the word is. You can choose from 4 options (Keys 1-4).\n\nBefore the starting, you will practice the task.\n\nPress SPACE to start the practice.\n",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    recall_instructions_key = keyboard.Keyboard(deviceName='recall_instructions_key')
    
    # --- Initialize components for Routine "iti_recall" ---
    ISI_2 = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI_2')
    
    # --- Initialize components for Routine "cued_recall" ---
    cue_stim_recall = visual.TextStim(win=win, name='cue_stim_recall',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    dash_stim_recall = visual.TextStim(win=win, name='dash_stim_recall',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    qmark_target = visual.TextStim(win=win, name='qmark_target',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    end_cued_recall = keyboard.Keyboard(deviceName='end_cued_recall')
    
    # --- Initialize components for Routine "recall_response" ---
    dash_stim_recall_resp = visual.TextStim(win=win, name='dash_stim_recall_resp',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_recall_resp = visual.TextStim(win=win, name='cue_stim_recall_resp',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    recall_stim_resp = visual.TextStim(win=win, name='recall_stim_resp',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    recall_question = visual.TextStim(win=win, name='recall_question',
        text='Could you recall the word pair?\n\nF) No        G) Yes',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    recall_reached = keyboard.Keyboard(deviceName='recall_reached')
    
    # --- Initialize components for Routine "recall_select" ---
    dash_stim_recall_select = visual.TextStim(win=win, name='dash_stim_recall_select',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    cue_stim_recall_select = visual.TextStim(win=win, name='cue_stim_recall_select',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    recall_stim_select = visual.TextStim(win=win, name='recall_stim_select',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    recall_question_select = visual.TextStim(win=win, name='recall_question_select',
        text='',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    recall_selection = keyboard.Keyboard(deviceName='recall_selection')
    
    # --- Initialize components for Routine "start_task" ---
    start_task_text = visual.TextStim(win=win, name='start_task_text',
        text='Now comes the task. \n\nPress SPACE when you are ready to start.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    start_key = keyboard.Keyboard(deviceName='start_key')
    
    # --- Initialize components for Routine "iti_recall" ---
    ISI_2 = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI_2')
    
    # --- Initialize components for Routine "cued_recall" ---
    cue_stim_recall = visual.TextStim(win=win, name='cue_stim_recall',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    dash_stim_recall = visual.TextStim(win=win, name='dash_stim_recall',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    qmark_target = visual.TextStim(win=win, name='qmark_target',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    end_cued_recall = keyboard.Keyboard(deviceName='end_cued_recall')
    
    # --- Initialize components for Routine "recall_response" ---
    dash_stim_recall_resp = visual.TextStim(win=win, name='dash_stim_recall_resp',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    cue_stim_recall_resp = visual.TextStim(win=win, name='cue_stim_recall_resp',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    recall_stim_resp = visual.TextStim(win=win, name='recall_stim_resp',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    recall_question = visual.TextStim(win=win, name='recall_question',
        text='Could you recall the word pair?\n\nF) No        G) Yes',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    recall_reached = keyboard.Keyboard(deviceName='recall_reached')
    
    # --- Initialize components for Routine "recall_select" ---
    dash_stim_recall_select = visual.TextStim(win=win, name='dash_stim_recall_select',
        text='-',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    cue_stim_recall_select = visual.TextStim(win=win, name='cue_stim_recall_select',
        text='',
        font='Arial',
        pos=(-0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    recall_stim_select = visual.TextStim(win=win, name='recall_stim_select',
        text='?',
        font='Arial',
        pos=(0.2, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    recall_question_select = visual.TextStim(win=win, name='recall_question_select',
        text='',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    recall_selection = keyboard.Keyboard(deviceName='recall_selection')
    
    # --- Initialize components for Routine "end_part2" ---
    end_part2_key = keyboard.Keyboard(deviceName='end_part2_key')
    end_part2_text = visual.TextStim(win=win, name='end_part2_text',
        text='You just finished Part 2 of the experiment!\n\nTake a 5 minute break before continuing.\n\nPress SPACE to continue ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "instructions_semantic_mapping" ---
    sm_instructions_text = visual.TextStim(win=win, name='sm_instructions_text',
        text='In the last part of the experiment, you will see a list of words, presented one by one on the screen. You will have to decide whether the meaning of the word refers to something living or non-living.\n\nIndicate your response with the F / G keys.\n\nPress SPACE to start the task.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sm_instructions_key = keyboard.Keyboard(deviceName='sm_instructions_key')
    
    # --- Initialize components for Routine "iti_mapping" ---
    ISI_3 = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='ISI_3')
    # Run 'Begin Experiment' code from mapping_trial_counter
    mapping_trial = 0
    
    # --- Initialize components for Routine "item" ---
    semantic_map_item = visual.TextStim(win=win, name='semantic_map_item',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    living_nonliving = keyboard.Keyboard(deviceName='living_nonliving')
    end_mapping = keyboard.Keyboard(deviceName='end_mapping')
    licing_nonliving_text = visual.TextStim(win=win, name='licing_nonliving_text',
        text='F) Living        G) Non-living',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    
    # --- Initialize components for Routine "mapping_break" ---
    break_text = visual.TextStim(win=win, name='break_text',
        text='Break. Press Space to continue.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    break_resp = keyboard.Keyboard(deviceName='break_resp')
    
    # --- Initialize components for Routine "thanks" ---
    thanks_text = visual.TextStim(win=win, name='thanks_text',
        text='This is the end of the experiment.\n\nThank you!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[welcome_text],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    thisExp.addData('welcome.started', welcome.tStart)
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 3.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
            # update params
            pass
        
        # if welcome_text is stopping this frame...
        if welcome_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > welcome_text.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                welcome_text.tStop = t  # not accounting for scr refresh
                welcome_text.tStopRefresh = tThisFlipGlobal  # on global time
                welcome_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'welcome_text.stopped')
                # update status
                welcome_text.status = FINISHED
                welcome_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome.stopped', welcome.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if welcome.maxDurationReached:
        routineTimer.addTime(-welcome.maxDuration)
    elif welcome.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-3.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "instructions1" ---
    # create an object to store info about Routine instructions1
    instructions1 = data.Routine(
        name='instructions1',
        components=[instructions1_text, instruction1_key],
    )
    instructions1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction1_key
    instruction1_key.keys = []
    instruction1_key.rt = []
    _instruction1_key_allKeys = []
    # store start times for instructions1
    instructions1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions1.tStart = globalClock.getTime(format='float')
    instructions1.status = STARTED
    thisExp.addData('instructions1.started', instructions1.tStart)
    instructions1.maxDuration = None
    # keep track of which components have finished
    instructions1Components = instructions1.components
    for thisComponent in instructions1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions1" ---
    instructions1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions1_text* updates
        
        # if instructions1_text is starting this frame...
        if instructions1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions1_text.frameNStart = frameN  # exact frame index
            instructions1_text.tStart = t  # local t and not account for scr refresh
            instructions1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions1_text.started')
            # update status
            instructions1_text.status = STARTED
            instructions1_text.setAutoDraw(True)
        
        # if instructions1_text is active this frame...
        if instructions1_text.status == STARTED:
            # update params
            pass
        
        # *instruction1_key* updates
        waitOnFlip = False
        
        # if instruction1_key is starting this frame...
        if instruction1_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction1_key.frameNStart = frameN  # exact frame index
            instruction1_key.tStart = t  # local t and not account for scr refresh
            instruction1_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction1_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction1_key.started')
            # update status
            instruction1_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction1_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction1_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction1_key.status == STARTED and not waitOnFlip:
            theseKeys = instruction1_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instruction1_key_allKeys.extend(theseKeys)
            if len(_instruction1_key_allKeys):
                instruction1_key.keys = _instruction1_key_allKeys[-1].name  # just the last key pressed
                instruction1_key.rt = _instruction1_key_allKeys[-1].rt
                instruction1_key.duration = _instruction1_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions1" ---
    for thisComponent in instructions1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions1
    instructions1.tStop = globalClock.getTime(format='float')
    instructions1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions1.stopped', instructions1.tStop)
    # check responses
    if instruction1_key.keys in ['', [], None]:  # No response was made
        instruction1_key.keys = None
    thisExp.addData('instruction1_key.keys',instruction1_key.keys)
    if instruction1_key.keys != None:  # we had a response
        thisExp.addData('instruction1_key.rt', instruction1_key.rt)
        thisExp.addData('instruction1_key.duration', instruction1_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions2" ---
    # create an object to store info about Routine instructions2
    instructions2 = data.Routine(
        name='instructions2',
        components=[instructions2_text, instruction2_key],
    )
    instructions2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction2_key
    instruction2_key.keys = []
    instruction2_key.rt = []
    _instruction2_key_allKeys = []
    # store start times for instructions2
    instructions2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions2.tStart = globalClock.getTime(format='float')
    instructions2.status = STARTED
    thisExp.addData('instructions2.started', instructions2.tStart)
    instructions2.maxDuration = None
    # keep track of which components have finished
    instructions2Components = instructions2.components
    for thisComponent in instructions2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions2" ---
    instructions2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions2_text* updates
        
        # if instructions2_text is starting this frame...
        if instructions2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions2_text.frameNStart = frameN  # exact frame index
            instructions2_text.tStart = t  # local t and not account for scr refresh
            instructions2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions2_text.started')
            # update status
            instructions2_text.status = STARTED
            instructions2_text.setAutoDraw(True)
        
        # if instructions2_text is active this frame...
        if instructions2_text.status == STARTED:
            # update params
            pass
        
        # *instruction2_key* updates
        waitOnFlip = False
        
        # if instruction2_key is starting this frame...
        if instruction2_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction2_key.frameNStart = frameN  # exact frame index
            instruction2_key.tStart = t  # local t and not account for scr refresh
            instruction2_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction2_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction2_key.started')
            # update status
            instruction2_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction2_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction2_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction2_key.status == STARTED and not waitOnFlip:
            theseKeys = instruction2_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instruction2_key_allKeys.extend(theseKeys)
            if len(_instruction2_key_allKeys):
                instruction2_key.keys = _instruction2_key_allKeys[-1].name  # just the last key pressed
                instruction2_key.rt = _instruction2_key_allKeys[-1].rt
                instruction2_key.duration = _instruction2_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions2" ---
    for thisComponent in instructions2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions2
    instructions2.tStop = globalClock.getTime(format='float')
    instructions2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions2.stopped', instructions2.tStop)
    # check responses
    if instruction2_key.keys in ['', [], None]:  # No response was made
        instruction2_key.keys = None
    thisExp.addData('instruction2_key.keys',instruction2_key.keys)
    if instruction2_key.keys != None:  # we had a response
        thisExp.addData('instruction2_key.rt', instruction2_key.rt)
        thisExp.addData('instruction2_key.duration', instruction2_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions3" ---
    # create an object to store info about Routine instructions3
    instructions3 = data.Routine(
        name='instructions3',
        components=[instructions3_text, instruction3_key],
    )
    instructions3.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction3_key
    instruction3_key.keys = []
    instruction3_key.rt = []
    _instruction3_key_allKeys = []
    # store start times for instructions3
    instructions3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions3.tStart = globalClock.getTime(format='float')
    instructions3.status = STARTED
    thisExp.addData('instructions3.started', instructions3.tStart)
    instructions3.maxDuration = None
    # keep track of which components have finished
    instructions3Components = instructions3.components
    for thisComponent in instructions3.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions3" ---
    instructions3.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions3_text* updates
        
        # if instructions3_text is starting this frame...
        if instructions3_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions3_text.frameNStart = frameN  # exact frame index
            instructions3_text.tStart = t  # local t and not account for scr refresh
            instructions3_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions3_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions3_text.started')
            # update status
            instructions3_text.status = STARTED
            instructions3_text.setAutoDraw(True)
        
        # if instructions3_text is active this frame...
        if instructions3_text.status == STARTED:
            # update params
            pass
        
        # *instruction3_key* updates
        waitOnFlip = False
        
        # if instruction3_key is starting this frame...
        if instruction3_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction3_key.frameNStart = frameN  # exact frame index
            instruction3_key.tStart = t  # local t and not account for scr refresh
            instruction3_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction3_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction3_key.started')
            # update status
            instruction3_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction3_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction3_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction3_key.status == STARTED and not waitOnFlip:
            theseKeys = instruction3_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instruction3_key_allKeys.extend(theseKeys)
            if len(_instruction3_key_allKeys):
                instruction3_key.keys = _instruction3_key_allKeys[-1].name  # just the last key pressed
                instruction3_key.rt = _instruction3_key_allKeys[-1].rt
                instruction3_key.duration = _instruction3_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions3.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions3.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions3" ---
    for thisComponent in instructions3.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions3
    instructions3.tStop = globalClock.getTime(format='float')
    instructions3.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions3.stopped', instructions3.tStop)
    # check responses
    if instruction3_key.keys in ['', [], None]:  # No response was made
        instruction3_key.keys = None
    thisExp.addData('instruction3_key.keys',instruction3_key.keys)
    if instruction3_key.keys != None:  # we had a response
        thisExp.addData('instruction3_key.rt', instruction3_key.rt)
        thisExp.addData('instruction3_key.duration', instruction3_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions3" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions4" ---
    # create an object to store info about Routine instructions4
    instructions4 = data.Routine(
        name='instructions4',
        components=[instructions4_text, instruction4_key],
    )
    instructions4.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instruction4_key
    instruction4_key.keys = []
    instruction4_key.rt = []
    _instruction4_key_allKeys = []
    # store start times for instructions4
    instructions4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions4.tStart = globalClock.getTime(format='float')
    instructions4.status = STARTED
    thisExp.addData('instructions4.started', instructions4.tStart)
    instructions4.maxDuration = None
    # keep track of which components have finished
    instructions4Components = instructions4.components
    for thisComponent in instructions4.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions4" ---
    instructions4.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions4_text* updates
        
        # if instructions4_text is starting this frame...
        if instructions4_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions4_text.frameNStart = frameN  # exact frame index
            instructions4_text.tStart = t  # local t and not account for scr refresh
            instructions4_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions4_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions4_text.started')
            # update status
            instructions4_text.status = STARTED
            instructions4_text.setAutoDraw(True)
        
        # if instructions4_text is active this frame...
        if instructions4_text.status == STARTED:
            # update params
            pass
        
        # *instruction4_key* updates
        waitOnFlip = False
        
        # if instruction4_key is starting this frame...
        if instruction4_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instruction4_key.frameNStart = frameN  # exact frame index
            instruction4_key.tStart = t  # local t and not account for scr refresh
            instruction4_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instruction4_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instruction4_key.started')
            # update status
            instruction4_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instruction4_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instruction4_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instruction4_key.status == STARTED and not waitOnFlip:
            theseKeys = instruction4_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instruction4_key_allKeys.extend(theseKeys)
            if len(_instruction4_key_allKeys):
                instruction4_key.keys = _instruction4_key_allKeys[-1].name  # just the last key pressed
                instruction4_key.rt = _instruction4_key_allKeys[-1].rt
                instruction4_key.duration = _instruction4_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions4.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions4.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions4" ---
    for thisComponent in instructions4.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions4
    instructions4.tStop = globalClock.getTime(format='float')
    instructions4.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions4.stopped', instructions4.tStop)
    # check responses
    if instruction4_key.keys in ['', [], None]:  # No response was made
        instruction4_key.keys = None
    thisExp.addData('instruction4_key.keys',instruction4_key.keys)
    if instruction4_key.keys != None:  # we had a response
        thisExp.addData('instruction4_key.rt', instruction4_key.rt)
        thisExp.addData('instruction4_key.duration', instruction4_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions4" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_trials = data.TrialHandler2(
        name='practice_trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('practice.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(practice_trials)  # add the loop to the experiment
    thisPractice_trial = practice_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
    if thisPractice_trial != None:
        for paramName in thisPractice_trial:
            globals()[paramName] = thisPractice_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_trial in practice_trials:
        currentLoop = practice_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
        if thisPractice_trial != None:
            for paramName in thisPractice_trial:
                globals()[paramName] = thisPractice_trial[paramName]
        
        # --- Prepare to start Routine "iti_learning_practice" ---
        # create an object to store info about Routine iti_learning_practice
        iti_learning_practice = data.Routine(
            name='iti_learning_practice',
            components=[ISI_4],
        )
        iti_learning_practice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from set_practice_guess_stim
        is_guess = practice_list.pop()
        # store start times for iti_learning_practice
        iti_learning_practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iti_learning_practice.tStart = globalClock.getTime(format='float')
        iti_learning_practice.status = STARTED
        thisExp.addData('iti_learning_practice.started', iti_learning_practice.tStart)
        iti_learning_practice.maxDuration = None
        # keep track of which components have finished
        iti_learning_practiceComponents = iti_learning_practice.components
        for thisComponent in iti_learning_practice.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iti_learning_practice" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        iti_learning_practice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *ISI_4* period
            
            # if ISI_4 is starting this frame...
            if ISI_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ISI_4.frameNStart = frameN  # exact frame index
                ISI_4.tStart = t  # local t and not account for scr refresh
                ISI_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ISI_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('ISI_4.started', t)
                # update status
                ISI_4.status = STARTED
                ISI_4.start(2.0)
            elif ISI_4.status == STARTED:  # one frame should pass before updating params and completing
                ISI_4.complete()  # finish the static period
                ISI_4.tStop = ISI_4.tStart + 2.0  # record stop time
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                iti_learning_practice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in iti_learning_practice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti_learning_practice" ---
        for thisComponent in iti_learning_practice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iti_learning_practice
        iti_learning_practice.tStop = globalClock.getTime(format='float')
        iti_learning_practice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iti_learning_practice.stopped', iti_learning_practice.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if iti_learning_practice.maxDurationReached:
            routineTimer.addTime(-iti_learning_practice.maxDuration)
        elif iti_learning_practice.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "guess" ---
        # create an object to store info about Routine guess
        guess = data.Routine(
            name='guess',
            components=[dash_stim_guessing, cue_stim_guessing, guess_stim, end_guess],
        )
        guess.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_guessing.setText(cue)
        # create starting attributes for end_guess
        end_guess.keys = []
        end_guess.rt = []
        _end_guess_allKeys = []
        # store start times for guess
        guess.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        guess.tStart = globalClock.getTime(format='float')
        guess.status = STARTED
        thisExp.addData('guess.started', guess.tStart)
        guess.maxDuration = None
        # skip Routine guess if its 'Skip if' condition is True
        guess.skipped = continueRoutine and not (is_guess != "Guess")
        continueRoutine = guess.skipped
        # keep track of which components have finished
        guessComponents = guess.components
        for thisComponent in guess.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "guess" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        guess.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_guessing* updates
            
            # if dash_stim_guessing is starting this frame...
            if dash_stim_guessing.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_guessing.frameNStart = frameN  # exact frame index
                dash_stim_guessing.tStart = t  # local t and not account for scr refresh
                dash_stim_guessing.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_guessing, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_guessing.started')
                # update status
                dash_stim_guessing.status = STARTED
                dash_stim_guessing.setAutoDraw(True)
            
            # if dash_stim_guessing is active this frame...
            if dash_stim_guessing.status == STARTED:
                # update params
                pass
            
            # if dash_stim_guessing is stopping this frame...
            if dash_stim_guessing.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_guessing.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_guessing.tStop = t  # not accounting for scr refresh
                    dash_stim_guessing.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_guessing.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_guessing.stopped')
                    # update status
                    dash_stim_guessing.status = FINISHED
                    dash_stim_guessing.setAutoDraw(False)
            
            # *cue_stim_guessing* updates
            
            # if cue_stim_guessing is starting this frame...
            if cue_stim_guessing.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_guessing.frameNStart = frameN  # exact frame index
                cue_stim_guessing.tStart = t  # local t and not account for scr refresh
                cue_stim_guessing.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_guessing, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_guessing.started')
                # update status
                cue_stim_guessing.status = STARTED
                cue_stim_guessing.setAutoDraw(True)
            
            # if cue_stim_guessing is active this frame...
            if cue_stim_guessing.status == STARTED:
                # update params
                pass
            
            # if cue_stim_guessing is stopping this frame...
            if cue_stim_guessing.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_guessing.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_guessing.tStop = t  # not accounting for scr refresh
                    cue_stim_guessing.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_guessing.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_guessing.stopped')
                    # update status
                    cue_stim_guessing.status = FINISHED
                    cue_stim_guessing.setAutoDraw(False)
            
            # *guess_stim* updates
            
            # if guess_stim is starting this frame...
            if guess_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_stim.frameNStart = frameN  # exact frame index
                guess_stim.tStart = t  # local t and not account for scr refresh
                guess_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_stim.started')
                # update status
                guess_stim.status = STARTED
                guess_stim.setAutoDraw(True)
            
            # if guess_stim is active this frame...
            if guess_stim.status == STARTED:
                # update params
                pass
            
            # if guess_stim is stopping this frame...
            if guess_stim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_stim.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_stim.tStop = t  # not accounting for scr refresh
                    guess_stim.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_stim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_stim.stopped')
                    # update status
                    guess_stim.status = FINISHED
                    guess_stim.setAutoDraw(False)
            
            # *end_guess* updates
            waitOnFlip = False
            
            # if end_guess is starting this frame...
            if end_guess.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_guess.frameNStart = frameN  # exact frame index
                end_guess.tStart = t  # local t and not account for scr refresh
                end_guess.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_guess, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_guess.started')
                # update status
                end_guess.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_guess.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_guess.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if end_guess is stopping this frame...
            if end_guess.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > end_guess.tStartRefresh + 6.00-frameTolerance:
                    # keep track of stop time/frame for later
                    end_guess.tStop = t  # not accounting for scr refresh
                    end_guess.tStopRefresh = tThisFlipGlobal  # on global time
                    end_guess.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_guess.stopped')
                    # update status
                    end_guess.status = FINISHED
                    end_guess.status = FINISHED
            if end_guess.status == STARTED and not waitOnFlip:
                theseKeys = end_guess.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _end_guess_allKeys.extend(theseKeys)
                if len(_end_guess_allKeys):
                    end_guess.keys = _end_guess_allKeys[-1].name  # just the last key pressed
                    end_guess.rt = _end_guess_allKeys[-1].rt
                    end_guess.duration = _end_guess_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                guess.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in guess.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "guess" ---
        for thisComponent in guess.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for guess
        guess.tStop = globalClock.getTime(format='float')
        guess.tStopRefresh = tThisFlipGlobal
        thisExp.addData('guess.stopped', guess.tStop)
        # check responses
        if end_guess.keys in ['', [], None]:  # No response was made
            end_guess.keys = None
        practice_trials.addData('end_guess.keys',end_guess.keys)
        if end_guess.keys != None:  # we had a response
            practice_trials.addData('end_guess.rt', end_guess.rt)
            practice_trials.addData('end_guess.duration', end_guess.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if guess.maxDurationReached:
            routineTimer.addTime(-guess.maxDuration)
        elif guess.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        
        # --- Prepare to start Routine "guess_response" ---
        # create an object to store info about Routine guess_response
        guess_response = data.Routine(
            name='guess_response',
            components=[dash_stim_guessing_resp, cue_stim_guessing_resp, guess_stim_resp, guess_question, guess_reached],
        )
        guess_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_guessing_resp.setText(cue)
        # create starting attributes for guess_reached
        guess_reached.keys = []
        guess_reached.rt = []
        _guess_reached_allKeys = []
        # store start times for guess_response
        guess_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        guess_response.tStart = globalClock.getTime(format='float')
        guess_response.status = STARTED
        thisExp.addData('guess_response.started', guess_response.tStart)
        guess_response.maxDuration = None
        # skip Routine guess_response if its 'Skip if' condition is True
        guess_response.skipped = continueRoutine and not (is_guess != "Guess")
        continueRoutine = guess_response.skipped
        # keep track of which components have finished
        guess_responseComponents = guess_response.components
        for thisComponent in guess_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "guess_response" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        guess_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_guessing_resp* updates
            
            # if dash_stim_guessing_resp is starting this frame...
            if dash_stim_guessing_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_guessing_resp.frameNStart = frameN  # exact frame index
                dash_stim_guessing_resp.tStart = t  # local t and not account for scr refresh
                dash_stim_guessing_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_guessing_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_guessing_resp.started')
                # update status
                dash_stim_guessing_resp.status = STARTED
                dash_stim_guessing_resp.setAutoDraw(True)
            
            # if dash_stim_guessing_resp is active this frame...
            if dash_stim_guessing_resp.status == STARTED:
                # update params
                pass
            
            # if dash_stim_guessing_resp is stopping this frame...
            if dash_stim_guessing_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_guessing_resp.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_guessing_resp.tStop = t  # not accounting for scr refresh
                    dash_stim_guessing_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_guessing_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_guessing_resp.stopped')
                    # update status
                    dash_stim_guessing_resp.status = FINISHED
                    dash_stim_guessing_resp.setAutoDraw(False)
            
            # *cue_stim_guessing_resp* updates
            
            # if cue_stim_guessing_resp is starting this frame...
            if cue_stim_guessing_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_guessing_resp.frameNStart = frameN  # exact frame index
                cue_stim_guessing_resp.tStart = t  # local t and not account for scr refresh
                cue_stim_guessing_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_guessing_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_guessing_resp.started')
                # update status
                cue_stim_guessing_resp.status = STARTED
                cue_stim_guessing_resp.setAutoDraw(True)
            
            # if cue_stim_guessing_resp is active this frame...
            if cue_stim_guessing_resp.status == STARTED:
                # update params
                pass
            
            # if cue_stim_guessing_resp is stopping this frame...
            if cue_stim_guessing_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_guessing_resp.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_guessing_resp.tStop = t  # not accounting for scr refresh
                    cue_stim_guessing_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_guessing_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_guessing_resp.stopped')
                    # update status
                    cue_stim_guessing_resp.status = FINISHED
                    cue_stim_guessing_resp.setAutoDraw(False)
            
            # *guess_stim_resp* updates
            
            # if guess_stim_resp is starting this frame...
            if guess_stim_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_stim_resp.frameNStart = frameN  # exact frame index
                guess_stim_resp.tStart = t  # local t and not account for scr refresh
                guess_stim_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_stim_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_stim_resp.started')
                # update status
                guess_stim_resp.status = STARTED
                guess_stim_resp.setAutoDraw(True)
            
            # if guess_stim_resp is active this frame...
            if guess_stim_resp.status == STARTED:
                # update params
                pass
            
            # if guess_stim_resp is stopping this frame...
            if guess_stim_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_stim_resp.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_stim_resp.tStop = t  # not accounting for scr refresh
                    guess_stim_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_stim_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_stim_resp.stopped')
                    # update status
                    guess_stim_resp.status = FINISHED
                    guess_stim_resp.setAutoDraw(False)
            
            # *guess_question* updates
            
            # if guess_question is starting this frame...
            if guess_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_question.frameNStart = frameN  # exact frame index
                guess_question.tStart = t  # local t and not account for scr refresh
                guess_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_question.started')
                # update status
                guess_question.status = STARTED
                guess_question.setAutoDraw(True)
            
            # if guess_question is active this frame...
            if guess_question.status == STARTED:
                # update params
                pass
            
            # if guess_question is stopping this frame...
            if guess_question.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_question.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_question.tStop = t  # not accounting for scr refresh
                    guess_question.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_question.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_question.stopped')
                    # update status
                    guess_question.status = FINISHED
                    guess_question.setAutoDraw(False)
            
            # *guess_reached* updates
            waitOnFlip = False
            
            # if guess_reached is starting this frame...
            if guess_reached.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_reached.frameNStart = frameN  # exact frame index
                guess_reached.tStart = t  # local t and not account for scr refresh
                guess_reached.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_reached, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_reached.started')
                # update status
                guess_reached.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(guess_reached.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(guess_reached.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if guess_reached is stopping this frame...
            if guess_reached.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_reached.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_reached.tStop = t  # not accounting for scr refresh
                    guess_reached.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_reached.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_reached.stopped')
                    # update status
                    guess_reached.status = FINISHED
                    guess_reached.status = FINISHED
            if guess_reached.status == STARTED and not waitOnFlip:
                theseKeys = guess_reached.getKeys(keyList=['f', 'g', 'h'], ignoreKeys=["escape"], waitRelease=False)
                _guess_reached_allKeys.extend(theseKeys)
                if len(_guess_reached_allKeys):
                    guess_reached.keys = _guess_reached_allKeys[-1].name  # just the last key pressed
                    guess_reached.rt = _guess_reached_allKeys[-1].rt
                    guess_reached.duration = _guess_reached_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                guess_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in guess_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "guess_response" ---
        for thisComponent in guess_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for guess_response
        guess_response.tStop = globalClock.getTime(format='float')
        guess_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('guess_response.stopped', guess_response.tStop)
        # check responses
        if guess_reached.keys in ['', [], None]:  # No response was made
            guess_reached.keys = None
        practice_trials.addData('guess_reached.keys',guess_reached.keys)
        if guess_reached.keys != None:  # we had a response
            practice_trials.addData('guess_reached.rt', guess_reached.rt)
            practice_trials.addData('guess_reached.duration', guess_reached.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if guess_response.maxDurationReached:
            routineTimer.addTime(-guess_response.maxDuration)
        elif guess_response.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "feedback_guess" ---
        # create an object to store info about Routine feedback_guess
        feedback_guess = data.Routine(
            name='feedback_guess',
            components=[no_response_feedback],
        )
        feedback_guess.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for feedback_guess
        feedback_guess.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback_guess.tStart = globalClock.getTime(format='float')
        feedback_guess.status = STARTED
        thisExp.addData('feedback_guess.started', feedback_guess.tStart)
        feedback_guess.maxDuration = None
        # skip Routine feedback_guess if its 'Skip if' condition is True
        feedback_guess.skipped = continueRoutine and not (is_guess != "Guess" or guess_reached.keys in ['f','g','h'])
        continueRoutine = feedback_guess.skipped
        # keep track of which components have finished
        feedback_guessComponents = feedback_guess.components
        for thisComponent in feedback_guess.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback_guess" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        feedback_guess.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *no_response_feedback* updates
            
            # if no_response_feedback is starting this frame...
            if no_response_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                no_response_feedback.frameNStart = frameN  # exact frame index
                no_response_feedback.tStart = t  # local t and not account for scr refresh
                no_response_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(no_response_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'no_response_feedback.started')
                # update status
                no_response_feedback.status = STARTED
                no_response_feedback.setAutoDraw(True)
            
            # if no_response_feedback is active this frame...
            if no_response_feedback.status == STARTED:
                # update params
                pass
            
            # if no_response_feedback is stopping this frame...
            if no_response_feedback.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > no_response_feedback.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    no_response_feedback.tStop = t  # not accounting for scr refresh
                    no_response_feedback.tStopRefresh = tThisFlipGlobal  # on global time
                    no_response_feedback.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'no_response_feedback.stopped')
                    # update status
                    no_response_feedback.status = FINISHED
                    no_response_feedback.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback_guess.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_guess.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_guess" ---
        for thisComponent in feedback_guess.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback_guess
        feedback_guess.tStop = globalClock.getTime(format='float')
        feedback_guess.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback_guess.stopped', feedback_guess.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback_guess.maxDurationReached:
            routineTimer.addTime(-feedback_guess.maxDuration)
        elif feedback_guess.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "encode" ---
        # create an object to store info about Routine encode
        encode = data.Routine(
            name='encode',
            components=[dash_stim_learning, cue_stim_learning, target_stim, end_encoding],
        )
        encode.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_learning.setText(cue)
        target_stim.setText(target)
        # create starting attributes for end_encoding
        end_encoding.keys = []
        end_encoding.rt = []
        _end_encoding_allKeys = []
        # store start times for encode
        encode.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        encode.tStart = globalClock.getTime(format='float')
        encode.status = STARTED
        thisExp.addData('encode.started', encode.tStart)
        encode.maxDuration = None
        # keep track of which components have finished
        encodeComponents = encode.components
        for thisComponent in encode.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "encode" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        encode.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_learning* updates
            
            # if dash_stim_learning is starting this frame...
            if dash_stim_learning.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_learning.frameNStart = frameN  # exact frame index
                dash_stim_learning.tStart = t  # local t and not account for scr refresh
                dash_stim_learning.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_learning, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_learning.started')
                # update status
                dash_stim_learning.status = STARTED
                dash_stim_learning.setAutoDraw(True)
            
            # if dash_stim_learning is active this frame...
            if dash_stim_learning.status == STARTED:
                # update params
                pass
            
            # if dash_stim_learning is stopping this frame...
            if dash_stim_learning.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_learning.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_learning.tStop = t  # not accounting for scr refresh
                    dash_stim_learning.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_learning.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_learning.stopped')
                    # update status
                    dash_stim_learning.status = FINISHED
                    dash_stim_learning.setAutoDraw(False)
            
            # *cue_stim_learning* updates
            
            # if cue_stim_learning is starting this frame...
            if cue_stim_learning.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_learning.frameNStart = frameN  # exact frame index
                cue_stim_learning.tStart = t  # local t and not account for scr refresh
                cue_stim_learning.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_learning, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_learning.started')
                # update status
                cue_stim_learning.status = STARTED
                cue_stim_learning.setAutoDraw(True)
            
            # if cue_stim_learning is active this frame...
            if cue_stim_learning.status == STARTED:
                # update params
                pass
            
            # if cue_stim_learning is stopping this frame...
            if cue_stim_learning.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_learning.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_learning.tStop = t  # not accounting for scr refresh
                    cue_stim_learning.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_learning.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_learning.stopped')
                    # update status
                    cue_stim_learning.status = FINISHED
                    cue_stim_learning.setAutoDraw(False)
            
            # *target_stim* updates
            
            # if target_stim is starting this frame...
            if target_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target_stim.frameNStart = frameN  # exact frame index
                target_stim.tStart = t  # local t and not account for scr refresh
                target_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target_stim.started')
                # update status
                target_stim.status = STARTED
                target_stim.setAutoDraw(True)
            
            # if target_stim is active this frame...
            if target_stim.status == STARTED:
                # update params
                pass
            
            # if target_stim is stopping this frame...
            if target_stim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target_stim.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    target_stim.tStop = t  # not accounting for scr refresh
                    target_stim.tStopRefresh = tThisFlipGlobal  # on global time
                    target_stim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_stim.stopped')
                    # update status
                    target_stim.status = FINISHED
                    target_stim.setAutoDraw(False)
            
            # *end_encoding* updates
            waitOnFlip = False
            
            # if end_encoding is starting this frame...
            if end_encoding.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_encoding.frameNStart = frameN  # exact frame index
                end_encoding.tStart = t  # local t and not account for scr refresh
                end_encoding.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_encoding, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_encoding.started')
                # update status
                end_encoding.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_encoding.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_encoding.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if end_encoding is stopping this frame...
            if end_encoding.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > end_encoding.tStartRefresh + 6.00-frameTolerance:
                    # keep track of stop time/frame for later
                    end_encoding.tStop = t  # not accounting for scr refresh
                    end_encoding.tStopRefresh = tThisFlipGlobal  # on global time
                    end_encoding.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_encoding.stopped')
                    # update status
                    end_encoding.status = FINISHED
                    end_encoding.status = FINISHED
            if end_encoding.status == STARTED and not waitOnFlip:
                theseKeys = end_encoding.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _end_encoding_allKeys.extend(theseKeys)
                if len(_end_encoding_allKeys):
                    end_encoding.keys = _end_encoding_allKeys[-1].name  # just the last key pressed
                    end_encoding.rt = _end_encoding_allKeys[-1].rt
                    end_encoding.duration = _end_encoding_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                encode.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in encode.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "encode" ---
        for thisComponent in encode.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for encode
        encode.tStop = globalClock.getTime(format='float')
        encode.tStopRefresh = tThisFlipGlobal
        thisExp.addData('encode.stopped', encode.tStop)
        # check responses
        if end_encoding.keys in ['', [], None]:  # No response was made
            end_encoding.keys = None
        practice_trials.addData('end_encoding.keys',end_encoding.keys)
        if end_encoding.keys != None:  # we had a response
            practice_trials.addData('end_encoding.rt', end_encoding.rt)
            practice_trials.addData('end_encoding.duration', end_encoding.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if encode.maxDurationReached:
            routineTimer.addTime(-encode.maxDuration)
        elif encode.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practice_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "start_task" ---
    # create an object to store info about Routine start_task
    start_task = data.Routine(
        name='start_task',
        components=[start_task_text, start_key],
    )
    start_task.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for start_key
    start_key.keys = []
    start_key.rt = []
    _start_key_allKeys = []
    # store start times for start_task
    start_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start_task.tStart = globalClock.getTime(format='float')
    start_task.status = STARTED
    thisExp.addData('start_task.started', start_task.tStart)
    start_task.maxDuration = None
    # keep track of which components have finished
    start_taskComponents = start_task.components
    for thisComponent in start_task.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_task" ---
    start_task.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_task_text* updates
        
        # if start_task_text is starting this frame...
        if start_task_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_task_text.frameNStart = frameN  # exact frame index
            start_task_text.tStart = t  # local t and not account for scr refresh
            start_task_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_task_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_task_text.started')
            # update status
            start_task_text.status = STARTED
            start_task_text.setAutoDraw(True)
        
        # if start_task_text is active this frame...
        if start_task_text.status == STARTED:
            # update params
            pass
        
        # *start_key* updates
        waitOnFlip = False
        
        # if start_key is starting this frame...
        if start_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_key.frameNStart = frameN  # exact frame index
            start_key.tStart = t  # local t and not account for scr refresh
            start_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_key.started')
            # update status
            start_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_key.status == STARTED and not waitOnFlip:
            theseKeys = start_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _start_key_allKeys.extend(theseKeys)
            if len(_start_key_allKeys):
                start_key.keys = _start_key_allKeys[-1].name  # just the last key pressed
                start_key.rt = _start_key_allKeys[-1].rt
                start_key.duration = _start_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start_task.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_task.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_task" ---
    for thisComponent in start_task.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start_task
    start_task.tStop = globalClock.getTime(format='float')
    start_task.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start_task.stopped', start_task.tStop)
    # check responses
    if start_key.keys in ['', [], None]:  # No response was made
        start_key.keys = None
    thisExp.addData('start_key.keys',start_key.keys)
    if start_key.keys != None:  # we had a response
        thisExp.addData('start_key.rt', start_key.rt)
        thisExp.addData('start_key.duration', start_key.duration)
    thisExp.nextEntry()
    # the Routine "start_task" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    learning_trials = data.TrialHandler2(
        name='learning_trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('cue_target_pairs.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(learning_trials)  # add the loop to the experiment
    thisLearning_trial = learning_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLearning_trial.rgb)
    if thisLearning_trial != None:
        for paramName in thisLearning_trial:
            globals()[paramName] = thisLearning_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisLearning_trial in learning_trials:
        currentLoop = learning_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisLearning_trial.rgb)
        if thisLearning_trial != None:
            for paramName in thisLearning_trial:
                globals()[paramName] = thisLearning_trial[paramName]
        
        # --- Prepare to start Routine "iti_learning" ---
        # create an object to store info about Routine iti_learning
        iti_learning = data.Routine(
            name='iti_learning',
            components=[ISI],
        )
        iti_learning.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from set_guess_stim
        is_guess = guess_list.pop()
        # store start times for iti_learning
        iti_learning.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iti_learning.tStart = globalClock.getTime(format='float')
        iti_learning.status = STARTED
        thisExp.addData('iti_learning.started', iti_learning.tStart)
        iti_learning.maxDuration = None
        # keep track of which components have finished
        iti_learningComponents = iti_learning.components
        for thisComponent in iti_learning.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iti_learning" ---
        # if trial has changed, end Routine now
        if isinstance(learning_trials, data.TrialHandler2) and thisLearning_trial.thisN != learning_trials.thisTrial.thisN:
            continueRoutine = False
        iti_learning.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *ISI* period
            
            # if ISI is starting this frame...
            if ISI.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ISI.frameNStart = frameN  # exact frame index
                ISI.tStart = t  # local t and not account for scr refresh
                ISI.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ISI, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('ISI.started', t)
                # update status
                ISI.status = STARTED
                ISI.start(2.0)
            elif ISI.status == STARTED:  # one frame should pass before updating params and completing
                ISI.complete()  # finish the static period
                ISI.tStop = ISI.tStart + 2.0  # record stop time
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                iti_learning.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in iti_learning.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti_learning" ---
        for thisComponent in iti_learning.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iti_learning
        iti_learning.tStop = globalClock.getTime(format='float')
        iti_learning.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iti_learning.stopped', iti_learning.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if iti_learning.maxDurationReached:
            routineTimer.addTime(-iti_learning.maxDuration)
        elif iti_learning.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "guess" ---
        # create an object to store info about Routine guess
        guess = data.Routine(
            name='guess',
            components=[dash_stim_guessing, cue_stim_guessing, guess_stim, end_guess],
        )
        guess.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_guessing.setText(cue)
        # create starting attributes for end_guess
        end_guess.keys = []
        end_guess.rt = []
        _end_guess_allKeys = []
        # store start times for guess
        guess.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        guess.tStart = globalClock.getTime(format='float')
        guess.status = STARTED
        thisExp.addData('guess.started', guess.tStart)
        guess.maxDuration = None
        # skip Routine guess if its 'Skip if' condition is True
        guess.skipped = continueRoutine and not (is_guess != "Guess")
        continueRoutine = guess.skipped
        # keep track of which components have finished
        guessComponents = guess.components
        for thisComponent in guess.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "guess" ---
        # if trial has changed, end Routine now
        if isinstance(learning_trials, data.TrialHandler2) and thisLearning_trial.thisN != learning_trials.thisTrial.thisN:
            continueRoutine = False
        guess.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_guessing* updates
            
            # if dash_stim_guessing is starting this frame...
            if dash_stim_guessing.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_guessing.frameNStart = frameN  # exact frame index
                dash_stim_guessing.tStart = t  # local t and not account for scr refresh
                dash_stim_guessing.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_guessing, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_guessing.started')
                # update status
                dash_stim_guessing.status = STARTED
                dash_stim_guessing.setAutoDraw(True)
            
            # if dash_stim_guessing is active this frame...
            if dash_stim_guessing.status == STARTED:
                # update params
                pass
            
            # if dash_stim_guessing is stopping this frame...
            if dash_stim_guessing.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_guessing.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_guessing.tStop = t  # not accounting for scr refresh
                    dash_stim_guessing.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_guessing.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_guessing.stopped')
                    # update status
                    dash_stim_guessing.status = FINISHED
                    dash_stim_guessing.setAutoDraw(False)
            
            # *cue_stim_guessing* updates
            
            # if cue_stim_guessing is starting this frame...
            if cue_stim_guessing.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_guessing.frameNStart = frameN  # exact frame index
                cue_stim_guessing.tStart = t  # local t and not account for scr refresh
                cue_stim_guessing.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_guessing, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_guessing.started')
                # update status
                cue_stim_guessing.status = STARTED
                cue_stim_guessing.setAutoDraw(True)
            
            # if cue_stim_guessing is active this frame...
            if cue_stim_guessing.status == STARTED:
                # update params
                pass
            
            # if cue_stim_guessing is stopping this frame...
            if cue_stim_guessing.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_guessing.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_guessing.tStop = t  # not accounting for scr refresh
                    cue_stim_guessing.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_guessing.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_guessing.stopped')
                    # update status
                    cue_stim_guessing.status = FINISHED
                    cue_stim_guessing.setAutoDraw(False)
            
            # *guess_stim* updates
            
            # if guess_stim is starting this frame...
            if guess_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_stim.frameNStart = frameN  # exact frame index
                guess_stim.tStart = t  # local t and not account for scr refresh
                guess_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_stim.started')
                # update status
                guess_stim.status = STARTED
                guess_stim.setAutoDraw(True)
            
            # if guess_stim is active this frame...
            if guess_stim.status == STARTED:
                # update params
                pass
            
            # if guess_stim is stopping this frame...
            if guess_stim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_stim.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_stim.tStop = t  # not accounting for scr refresh
                    guess_stim.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_stim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_stim.stopped')
                    # update status
                    guess_stim.status = FINISHED
                    guess_stim.setAutoDraw(False)
            
            # *end_guess* updates
            waitOnFlip = False
            
            # if end_guess is starting this frame...
            if end_guess.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_guess.frameNStart = frameN  # exact frame index
                end_guess.tStart = t  # local t and not account for scr refresh
                end_guess.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_guess, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_guess.started')
                # update status
                end_guess.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_guess.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_guess.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if end_guess is stopping this frame...
            if end_guess.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > end_guess.tStartRefresh + 6.00-frameTolerance:
                    # keep track of stop time/frame for later
                    end_guess.tStop = t  # not accounting for scr refresh
                    end_guess.tStopRefresh = tThisFlipGlobal  # on global time
                    end_guess.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_guess.stopped')
                    # update status
                    end_guess.status = FINISHED
                    end_guess.status = FINISHED
            if end_guess.status == STARTED and not waitOnFlip:
                theseKeys = end_guess.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _end_guess_allKeys.extend(theseKeys)
                if len(_end_guess_allKeys):
                    end_guess.keys = _end_guess_allKeys[-1].name  # just the last key pressed
                    end_guess.rt = _end_guess_allKeys[-1].rt
                    end_guess.duration = _end_guess_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                guess.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in guess.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "guess" ---
        for thisComponent in guess.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for guess
        guess.tStop = globalClock.getTime(format='float')
        guess.tStopRefresh = tThisFlipGlobal
        thisExp.addData('guess.stopped', guess.tStop)
        # check responses
        if end_guess.keys in ['', [], None]:  # No response was made
            end_guess.keys = None
        learning_trials.addData('end_guess.keys',end_guess.keys)
        if end_guess.keys != None:  # we had a response
            learning_trials.addData('end_guess.rt', end_guess.rt)
            learning_trials.addData('end_guess.duration', end_guess.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if guess.maxDurationReached:
            routineTimer.addTime(-guess.maxDuration)
        elif guess.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        
        # --- Prepare to start Routine "guess_response" ---
        # create an object to store info about Routine guess_response
        guess_response = data.Routine(
            name='guess_response',
            components=[dash_stim_guessing_resp, cue_stim_guessing_resp, guess_stim_resp, guess_question, guess_reached],
        )
        guess_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_guessing_resp.setText(cue)
        # create starting attributes for guess_reached
        guess_reached.keys = []
        guess_reached.rt = []
        _guess_reached_allKeys = []
        # store start times for guess_response
        guess_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        guess_response.tStart = globalClock.getTime(format='float')
        guess_response.status = STARTED
        thisExp.addData('guess_response.started', guess_response.tStart)
        guess_response.maxDuration = None
        # skip Routine guess_response if its 'Skip if' condition is True
        guess_response.skipped = continueRoutine and not (is_guess != "Guess")
        continueRoutine = guess_response.skipped
        # keep track of which components have finished
        guess_responseComponents = guess_response.components
        for thisComponent in guess_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "guess_response" ---
        # if trial has changed, end Routine now
        if isinstance(learning_trials, data.TrialHandler2) and thisLearning_trial.thisN != learning_trials.thisTrial.thisN:
            continueRoutine = False
        guess_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_guessing_resp* updates
            
            # if dash_stim_guessing_resp is starting this frame...
            if dash_stim_guessing_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_guessing_resp.frameNStart = frameN  # exact frame index
                dash_stim_guessing_resp.tStart = t  # local t and not account for scr refresh
                dash_stim_guessing_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_guessing_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_guessing_resp.started')
                # update status
                dash_stim_guessing_resp.status = STARTED
                dash_stim_guessing_resp.setAutoDraw(True)
            
            # if dash_stim_guessing_resp is active this frame...
            if dash_stim_guessing_resp.status == STARTED:
                # update params
                pass
            
            # if dash_stim_guessing_resp is stopping this frame...
            if dash_stim_guessing_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_guessing_resp.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_guessing_resp.tStop = t  # not accounting for scr refresh
                    dash_stim_guessing_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_guessing_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_guessing_resp.stopped')
                    # update status
                    dash_stim_guessing_resp.status = FINISHED
                    dash_stim_guessing_resp.setAutoDraw(False)
            
            # *cue_stim_guessing_resp* updates
            
            # if cue_stim_guessing_resp is starting this frame...
            if cue_stim_guessing_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_guessing_resp.frameNStart = frameN  # exact frame index
                cue_stim_guessing_resp.tStart = t  # local t and not account for scr refresh
                cue_stim_guessing_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_guessing_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_guessing_resp.started')
                # update status
                cue_stim_guessing_resp.status = STARTED
                cue_stim_guessing_resp.setAutoDraw(True)
            
            # if cue_stim_guessing_resp is active this frame...
            if cue_stim_guessing_resp.status == STARTED:
                # update params
                pass
            
            # if cue_stim_guessing_resp is stopping this frame...
            if cue_stim_guessing_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_guessing_resp.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_guessing_resp.tStop = t  # not accounting for scr refresh
                    cue_stim_guessing_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_guessing_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_guessing_resp.stopped')
                    # update status
                    cue_stim_guessing_resp.status = FINISHED
                    cue_stim_guessing_resp.setAutoDraw(False)
            
            # *guess_stim_resp* updates
            
            # if guess_stim_resp is starting this frame...
            if guess_stim_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_stim_resp.frameNStart = frameN  # exact frame index
                guess_stim_resp.tStart = t  # local t and not account for scr refresh
                guess_stim_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_stim_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_stim_resp.started')
                # update status
                guess_stim_resp.status = STARTED
                guess_stim_resp.setAutoDraw(True)
            
            # if guess_stim_resp is active this frame...
            if guess_stim_resp.status == STARTED:
                # update params
                pass
            
            # if guess_stim_resp is stopping this frame...
            if guess_stim_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_stim_resp.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_stim_resp.tStop = t  # not accounting for scr refresh
                    guess_stim_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_stim_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_stim_resp.stopped')
                    # update status
                    guess_stim_resp.status = FINISHED
                    guess_stim_resp.setAutoDraw(False)
            
            # *guess_question* updates
            
            # if guess_question is starting this frame...
            if guess_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_question.frameNStart = frameN  # exact frame index
                guess_question.tStart = t  # local t and not account for scr refresh
                guess_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_question.started')
                # update status
                guess_question.status = STARTED
                guess_question.setAutoDraw(True)
            
            # if guess_question is active this frame...
            if guess_question.status == STARTED:
                # update params
                pass
            
            # if guess_question is stopping this frame...
            if guess_question.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_question.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_question.tStop = t  # not accounting for scr refresh
                    guess_question.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_question.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_question.stopped')
                    # update status
                    guess_question.status = FINISHED
                    guess_question.setAutoDraw(False)
            
            # *guess_reached* updates
            waitOnFlip = False
            
            # if guess_reached is starting this frame...
            if guess_reached.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                guess_reached.frameNStart = frameN  # exact frame index
                guess_reached.tStart = t  # local t and not account for scr refresh
                guess_reached.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(guess_reached, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'guess_reached.started')
                # update status
                guess_reached.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(guess_reached.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(guess_reached.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if guess_reached is stopping this frame...
            if guess_reached.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > guess_reached.tStartRefresh + 3.0-frameTolerance:
                    # keep track of stop time/frame for later
                    guess_reached.tStop = t  # not accounting for scr refresh
                    guess_reached.tStopRefresh = tThisFlipGlobal  # on global time
                    guess_reached.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'guess_reached.stopped')
                    # update status
                    guess_reached.status = FINISHED
                    guess_reached.status = FINISHED
            if guess_reached.status == STARTED and not waitOnFlip:
                theseKeys = guess_reached.getKeys(keyList=['f', 'g', 'h'], ignoreKeys=["escape"], waitRelease=False)
                _guess_reached_allKeys.extend(theseKeys)
                if len(_guess_reached_allKeys):
                    guess_reached.keys = _guess_reached_allKeys[-1].name  # just the last key pressed
                    guess_reached.rt = _guess_reached_allKeys[-1].rt
                    guess_reached.duration = _guess_reached_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                guess_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in guess_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "guess_response" ---
        for thisComponent in guess_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for guess_response
        guess_response.tStop = globalClock.getTime(format='float')
        guess_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('guess_response.stopped', guess_response.tStop)
        # check responses
        if guess_reached.keys in ['', [], None]:  # No response was made
            guess_reached.keys = None
        learning_trials.addData('guess_reached.keys',guess_reached.keys)
        if guess_reached.keys != None:  # we had a response
            learning_trials.addData('guess_reached.rt', guess_reached.rt)
            learning_trials.addData('guess_reached.duration', guess_reached.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if guess_response.maxDurationReached:
            routineTimer.addTime(-guess_response.maxDuration)
        elif guess_response.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        
        # --- Prepare to start Routine "encode" ---
        # create an object to store info about Routine encode
        encode = data.Routine(
            name='encode',
            components=[dash_stim_learning, cue_stim_learning, target_stim, end_encoding],
        )
        encode.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_learning.setText(cue)
        target_stim.setText(target)
        # create starting attributes for end_encoding
        end_encoding.keys = []
        end_encoding.rt = []
        _end_encoding_allKeys = []
        # store start times for encode
        encode.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        encode.tStart = globalClock.getTime(format='float')
        encode.status = STARTED
        thisExp.addData('encode.started', encode.tStart)
        encode.maxDuration = None
        # keep track of which components have finished
        encodeComponents = encode.components
        for thisComponent in encode.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "encode" ---
        # if trial has changed, end Routine now
        if isinstance(learning_trials, data.TrialHandler2) and thisLearning_trial.thisN != learning_trials.thisTrial.thisN:
            continueRoutine = False
        encode.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 6.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_learning* updates
            
            # if dash_stim_learning is starting this frame...
            if dash_stim_learning.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_learning.frameNStart = frameN  # exact frame index
                dash_stim_learning.tStart = t  # local t and not account for scr refresh
                dash_stim_learning.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_learning, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_learning.started')
                # update status
                dash_stim_learning.status = STARTED
                dash_stim_learning.setAutoDraw(True)
            
            # if dash_stim_learning is active this frame...
            if dash_stim_learning.status == STARTED:
                # update params
                pass
            
            # if dash_stim_learning is stopping this frame...
            if dash_stim_learning.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_learning.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_learning.tStop = t  # not accounting for scr refresh
                    dash_stim_learning.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_learning.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_learning.stopped')
                    # update status
                    dash_stim_learning.status = FINISHED
                    dash_stim_learning.setAutoDraw(False)
            
            # *cue_stim_learning* updates
            
            # if cue_stim_learning is starting this frame...
            if cue_stim_learning.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_learning.frameNStart = frameN  # exact frame index
                cue_stim_learning.tStart = t  # local t and not account for scr refresh
                cue_stim_learning.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_learning, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_learning.started')
                # update status
                cue_stim_learning.status = STARTED
                cue_stim_learning.setAutoDraw(True)
            
            # if cue_stim_learning is active this frame...
            if cue_stim_learning.status == STARTED:
                # update params
                pass
            
            # if cue_stim_learning is stopping this frame...
            if cue_stim_learning.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_learning.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_learning.tStop = t  # not accounting for scr refresh
                    cue_stim_learning.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_learning.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_learning.stopped')
                    # update status
                    cue_stim_learning.status = FINISHED
                    cue_stim_learning.setAutoDraw(False)
            
            # *target_stim* updates
            
            # if target_stim is starting this frame...
            if target_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                target_stim.frameNStart = frameN  # exact frame index
                target_stim.tStart = t  # local t and not account for scr refresh
                target_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(target_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'target_stim.started')
                # update status
                target_stim.status = STARTED
                target_stim.setAutoDraw(True)
            
            # if target_stim is active this frame...
            if target_stim.status == STARTED:
                # update params
                pass
            
            # if target_stim is stopping this frame...
            if target_stim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > target_stim.tStartRefresh + 6.0-frameTolerance:
                    # keep track of stop time/frame for later
                    target_stim.tStop = t  # not accounting for scr refresh
                    target_stim.tStopRefresh = tThisFlipGlobal  # on global time
                    target_stim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'target_stim.stopped')
                    # update status
                    target_stim.status = FINISHED
                    target_stim.setAutoDraw(False)
            
            # *end_encoding* updates
            waitOnFlip = False
            
            # if end_encoding is starting this frame...
            if end_encoding.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_encoding.frameNStart = frameN  # exact frame index
                end_encoding.tStart = t  # local t and not account for scr refresh
                end_encoding.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_encoding, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_encoding.started')
                # update status
                end_encoding.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_encoding.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_encoding.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if end_encoding is stopping this frame...
            if end_encoding.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > end_encoding.tStartRefresh + 6.00-frameTolerance:
                    # keep track of stop time/frame for later
                    end_encoding.tStop = t  # not accounting for scr refresh
                    end_encoding.tStopRefresh = tThisFlipGlobal  # on global time
                    end_encoding.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_encoding.stopped')
                    # update status
                    end_encoding.status = FINISHED
                    end_encoding.status = FINISHED
            if end_encoding.status == STARTED and not waitOnFlip:
                theseKeys = end_encoding.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _end_encoding_allKeys.extend(theseKeys)
                if len(_end_encoding_allKeys):
                    end_encoding.keys = _end_encoding_allKeys[-1].name  # just the last key pressed
                    end_encoding.rt = _end_encoding_allKeys[-1].rt
                    end_encoding.duration = _end_encoding_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                encode.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in encode.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "encode" ---
        for thisComponent in encode.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for encode
        encode.tStop = globalClock.getTime(format='float')
        encode.tStopRefresh = tThisFlipGlobal
        thisExp.addData('encode.stopped', encode.tStop)
        # check responses
        if end_encoding.keys in ['', [], None]:  # No response was made
            end_encoding.keys = None
        learning_trials.addData('end_encoding.keys',end_encoding.keys)
        if end_encoding.keys != None:  # we had a response
            learning_trials.addData('end_encoding.rt', end_encoding.rt)
            learning_trials.addData('end_encoding.duration', end_encoding.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if encode.maxDurationReached:
            routineTimer.addTime(-encode.maxDuration)
        elif encode.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-6.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'learning_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end_part1" ---
    # create an object to store info about Routine end_part1
    end_part1 = data.Routine(
        name='end_part1',
        components=[end_part1_key, end_part1_text],
    )
    end_part1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for end_part1_key
    end_part1_key.keys = []
    end_part1_key.rt = []
    _end_part1_key_allKeys = []
    # store start times for end_part1
    end_part1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_part1.tStart = globalClock.getTime(format='float')
    end_part1.status = STARTED
    thisExp.addData('end_part1.started', end_part1.tStart)
    end_part1.maxDuration = None
    # keep track of which components have finished
    end_part1Components = end_part1.components
    for thisComponent in end_part1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_part1" ---
    end_part1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_part1_key* updates
        waitOnFlip = False
        
        # if end_part1_key is starting this frame...
        if end_part1_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_part1_key.frameNStart = frameN  # exact frame index
            end_part1_key.tStart = t  # local t and not account for scr refresh
            end_part1_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_part1_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_part1_key.started')
            # update status
            end_part1_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_part1_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_part1_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_part1_key.status == STARTED and not waitOnFlip:
            theseKeys = end_part1_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _end_part1_key_allKeys.extend(theseKeys)
            if len(_end_part1_key_allKeys):
                end_part1_key.keys = _end_part1_key_allKeys[-1].name  # just the last key pressed
                end_part1_key.rt = _end_part1_key_allKeys[-1].rt
                end_part1_key.duration = _end_part1_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *end_part1_text* updates
        
        # if end_part1_text is starting this frame...
        if end_part1_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_part1_text.frameNStart = frameN  # exact frame index
            end_part1_text.tStart = t  # local t and not account for scr refresh
            end_part1_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_part1_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_part1_text.started')
            # update status
            end_part1_text.status = STARTED
            end_part1_text.setAutoDraw(True)
        
        # if end_part1_text is active this frame...
        if end_part1_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end_part1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_part1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_part1" ---
    for thisComponent in end_part1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_part1
    end_part1.tStop = globalClock.getTime(format='float')
    end_part1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_part1.stopped', end_part1.tStop)
    # check responses
    if end_part1_key.keys in ['', [], None]:  # No response was made
        end_part1_key.keys = None
    thisExp.addData('end_part1_key.keys',end_part1_key.keys)
    if end_part1_key.keys != None:  # we had a response
        thisExp.addData('end_part1_key.rt', end_part1_key.rt)
        thisExp.addData('end_part1_key.duration', end_part1_key.duration)
    thisExp.nextEntry()
    # the Routine "end_part1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_recall" ---
    # create an object to store info about Routine instructions_recall
    instructions_recall = data.Routine(
        name='instructions_recall',
        components=[recall_instructions_text, recall_instructions_key],
    )
    instructions_recall.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for recall_instructions_key
    recall_instructions_key.keys = []
    recall_instructions_key.rt = []
    _recall_instructions_key_allKeys = []
    # store start times for instructions_recall
    instructions_recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_recall.tStart = globalClock.getTime(format='float')
    instructions_recall.status = STARTED
    thisExp.addData('instructions_recall.started', instructions_recall.tStart)
    instructions_recall.maxDuration = None
    # keep track of which components have finished
    instructions_recallComponents = instructions_recall.components
    for thisComponent in instructions_recall.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_recall" ---
    instructions_recall.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *recall_instructions_text* updates
        
        # if recall_instructions_text is starting this frame...
        if recall_instructions_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            recall_instructions_text.frameNStart = frameN  # exact frame index
            recall_instructions_text.tStart = t  # local t and not account for scr refresh
            recall_instructions_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(recall_instructions_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'recall_instructions_text.started')
            # update status
            recall_instructions_text.status = STARTED
            recall_instructions_text.setAutoDraw(True)
        
        # if recall_instructions_text is active this frame...
        if recall_instructions_text.status == STARTED:
            # update params
            pass
        
        # *recall_instructions_key* updates
        waitOnFlip = False
        
        # if recall_instructions_key is starting this frame...
        if recall_instructions_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            recall_instructions_key.frameNStart = frameN  # exact frame index
            recall_instructions_key.tStart = t  # local t and not account for scr refresh
            recall_instructions_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(recall_instructions_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'recall_instructions_key.started')
            # update status
            recall_instructions_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(recall_instructions_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(recall_instructions_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if recall_instructions_key.status == STARTED and not waitOnFlip:
            theseKeys = recall_instructions_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _recall_instructions_key_allKeys.extend(theseKeys)
            if len(_recall_instructions_key_allKeys):
                recall_instructions_key.keys = _recall_instructions_key_allKeys[-1].name  # just the last key pressed
                recall_instructions_key.rt = _recall_instructions_key_allKeys[-1].rt
                recall_instructions_key.duration = _recall_instructions_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_recall.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_recall.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_recall" ---
    for thisComponent in instructions_recall.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_recall
    instructions_recall.tStop = globalClock.getTime(format='float')
    instructions_recall.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_recall.stopped', instructions_recall.tStop)
    # check responses
    if recall_instructions_key.keys in ['', [], None]:  # No response was made
        recall_instructions_key.keys = None
    thisExp.addData('recall_instructions_key.keys',recall_instructions_key.keys)
    if recall_instructions_key.keys != None:  # we had a response
        thisExp.addData('recall_instructions_key.rt', recall_instructions_key.rt)
        thisExp.addData('recall_instructions_key.duration', recall_instructions_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions_recall" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    test_practice_trials = data.TrialHandler2(
        name='test_practice_trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('practice.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(test_practice_trials)  # add the loop to the experiment
    thisTest_practice_trial = test_practice_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTest_practice_trial.rgb)
    if thisTest_practice_trial != None:
        for paramName in thisTest_practice_trial:
            globals()[paramName] = thisTest_practice_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTest_practice_trial in test_practice_trials:
        currentLoop = test_practice_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTest_practice_trial.rgb)
        if thisTest_practice_trial != None:
            for paramName in thisTest_practice_trial:
                globals()[paramName] = thisTest_practice_trial[paramName]
        
        # --- Prepare to start Routine "iti_recall" ---
        # create an object to store info about Routine iti_recall
        iti_recall = data.Routine(
            name='iti_recall',
            components=[ISI_2],
        )
        iti_recall.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for iti_recall
        iti_recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iti_recall.tStart = globalClock.getTime(format='float')
        iti_recall.status = STARTED
        thisExp.addData('iti_recall.started', iti_recall.tStart)
        iti_recall.maxDuration = None
        # keep track of which components have finished
        iti_recallComponents = iti_recall.components
        for thisComponent in iti_recall.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iti_recall" ---
        # if trial has changed, end Routine now
        if isinstance(test_practice_trials, data.TrialHandler2) and thisTest_practice_trial.thisN != test_practice_trials.thisTrial.thisN:
            continueRoutine = False
        iti_recall.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *ISI_2* period
            
            # if ISI_2 is starting this frame...
            if ISI_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ISI_2.frameNStart = frameN  # exact frame index
                ISI_2.tStart = t  # local t and not account for scr refresh
                ISI_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ISI_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('ISI_2.started', t)
                # update status
                ISI_2.status = STARTED
                ISI_2.start(2.0)
            elif ISI_2.status == STARTED:  # one frame should pass before updating params and completing
                ISI_2.complete()  # finish the static period
                ISI_2.tStop = ISI_2.tStart + 2.0  # record stop time
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                iti_recall.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in iti_recall.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti_recall" ---
        for thisComponent in iti_recall.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iti_recall
        iti_recall.tStop = globalClock.getTime(format='float')
        iti_recall.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iti_recall.stopped', iti_recall.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if iti_recall.maxDurationReached:
            routineTimer.addTime(-iti_recall.maxDuration)
        elif iti_recall.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "cued_recall" ---
        # create an object to store info about Routine cued_recall
        cued_recall = data.Routine(
            name='cued_recall',
            components=[cue_stim_recall, dash_stim_recall, qmark_target, end_cued_recall],
        )
        cued_recall.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_recall.setText(cue)
        # create starting attributes for end_cued_recall
        end_cued_recall.keys = []
        end_cued_recall.rt = []
        _end_cued_recall_allKeys = []
        # store start times for cued_recall
        cued_recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        cued_recall.tStart = globalClock.getTime(format='float')
        cued_recall.status = STARTED
        thisExp.addData('cued_recall.started', cued_recall.tStart)
        cued_recall.maxDuration = None
        # keep track of which components have finished
        cued_recallComponents = cued_recall.components
        for thisComponent in cued_recall.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "cued_recall" ---
        # if trial has changed, end Routine now
        if isinstance(test_practice_trials, data.TrialHandler2) and thisTest_practice_trial.thisN != test_practice_trials.thisTrial.thisN:
            continueRoutine = False
        cued_recall.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cue_stim_recall* updates
            
            # if cue_stim_recall is starting this frame...
            if cue_stim_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_recall.frameNStart = frameN  # exact frame index
                cue_stim_recall.tStart = t  # local t and not account for scr refresh
                cue_stim_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_recall, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_recall.started')
                # update status
                cue_stim_recall.status = STARTED
                cue_stim_recall.setAutoDraw(True)
            
            # if cue_stim_recall is active this frame...
            if cue_stim_recall.status == STARTED:
                # update params
                pass
            
            # if cue_stim_recall is stopping this frame...
            if cue_stim_recall.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_recall.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_recall.tStop = t  # not accounting for scr refresh
                    cue_stim_recall.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_recall.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall.stopped')
                    # update status
                    cue_stim_recall.status = FINISHED
                    cue_stim_recall.setAutoDraw(False)
            
            # *dash_stim_recall* updates
            
            # if dash_stim_recall is starting this frame...
            if dash_stim_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_recall.frameNStart = frameN  # exact frame index
                dash_stim_recall.tStart = t  # local t and not account for scr refresh
                dash_stim_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_recall, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_recall.started')
                # update status
                dash_stim_recall.status = STARTED
                dash_stim_recall.setAutoDraw(True)
            
            # if dash_stim_recall is active this frame...
            if dash_stim_recall.status == STARTED:
                # update params
                pass
            
            # if dash_stim_recall is stopping this frame...
            if dash_stim_recall.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_recall.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_recall.tStop = t  # not accounting for scr refresh
                    dash_stim_recall.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_recall.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall.stopped')
                    # update status
                    dash_stim_recall.status = FINISHED
                    dash_stim_recall.setAutoDraw(False)
            
            # *qmark_target* updates
            
            # if qmark_target is starting this frame...
            if qmark_target.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                qmark_target.frameNStart = frameN  # exact frame index
                qmark_target.tStart = t  # local t and not account for scr refresh
                qmark_target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(qmark_target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'qmark_target.started')
                # update status
                qmark_target.status = STARTED
                qmark_target.setAutoDraw(True)
            
            # if qmark_target is active this frame...
            if qmark_target.status == STARTED:
                # update params
                pass
            
            # if qmark_target is stopping this frame...
            if qmark_target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > qmark_target.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    qmark_target.tStop = t  # not accounting for scr refresh
                    qmark_target.tStopRefresh = tThisFlipGlobal  # on global time
                    qmark_target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'qmark_target.stopped')
                    # update status
                    qmark_target.status = FINISHED
                    qmark_target.setAutoDraw(False)
            
            # *end_cued_recall* updates
            waitOnFlip = False
            
            # if end_cued_recall is starting this frame...
            if end_cued_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_cued_recall.frameNStart = frameN  # exact frame index
                end_cued_recall.tStart = t  # local t and not account for scr refresh
                end_cued_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_cued_recall, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_cued_recall.started')
                # update status
                end_cued_recall.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_cued_recall.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_cued_recall.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if end_cued_recall is stopping this frame...
            if end_cued_recall.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > end_cued_recall.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    end_cued_recall.tStop = t  # not accounting for scr refresh
                    end_cued_recall.tStopRefresh = tThisFlipGlobal  # on global time
                    end_cued_recall.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_cued_recall.stopped')
                    # update status
                    end_cued_recall.status = FINISHED
                    end_cued_recall.status = FINISHED
            if end_cued_recall.status == STARTED and not waitOnFlip:
                theseKeys = end_cued_recall.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _end_cued_recall_allKeys.extend(theseKeys)
                if len(_end_cued_recall_allKeys):
                    end_cued_recall.keys = _end_cued_recall_allKeys[-1].name  # just the last key pressed
                    end_cued_recall.rt = _end_cued_recall_allKeys[-1].rt
                    end_cued_recall.duration = _end_cued_recall_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                cued_recall.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in cued_recall.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "cued_recall" ---
        for thisComponent in cued_recall.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for cued_recall
        cued_recall.tStop = globalClock.getTime(format='float')
        cued_recall.tStopRefresh = tThisFlipGlobal
        thisExp.addData('cued_recall.stopped', cued_recall.tStop)
        # check responses
        if end_cued_recall.keys in ['', [], None]:  # No response was made
            end_cued_recall.keys = None
        test_practice_trials.addData('end_cued_recall.keys',end_cued_recall.keys)
        if end_cued_recall.keys != None:  # we had a response
            test_practice_trials.addData('end_cued_recall.rt', end_cued_recall.rt)
            test_practice_trials.addData('end_cued_recall.duration', end_cued_recall.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if cued_recall.maxDurationReached:
            routineTimer.addTime(-cued_recall.maxDuration)
        elif cued_recall.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        
        # --- Prepare to start Routine "recall_response" ---
        # create an object to store info about Routine recall_response
        recall_response = data.Routine(
            name='recall_response',
            components=[dash_stim_recall_resp, cue_stim_recall_resp, recall_stim_resp, recall_question, recall_reached],
        )
        recall_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_recall_resp.setText(cue)
        # create starting attributes for recall_reached
        recall_reached.keys = []
        recall_reached.rt = []
        _recall_reached_allKeys = []
        # store start times for recall_response
        recall_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_response.tStart = globalClock.getTime(format='float')
        recall_response.status = STARTED
        thisExp.addData('recall_response.started', recall_response.tStart)
        recall_response.maxDuration = None
        # keep track of which components have finished
        recall_responseComponents = recall_response.components
        for thisComponent in recall_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_response" ---
        # if trial has changed, end Routine now
        if isinstance(test_practice_trials, data.TrialHandler2) and thisTest_practice_trial.thisN != test_practice_trials.thisTrial.thisN:
            continueRoutine = False
        recall_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_recall_resp* updates
            
            # if dash_stim_recall_resp is starting this frame...
            if dash_stim_recall_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_recall_resp.frameNStart = frameN  # exact frame index
                dash_stim_recall_resp.tStart = t  # local t and not account for scr refresh
                dash_stim_recall_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_recall_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_recall_resp.started')
                # update status
                dash_stim_recall_resp.status = STARTED
                dash_stim_recall_resp.setAutoDraw(True)
            
            # if dash_stim_recall_resp is active this frame...
            if dash_stim_recall_resp.status == STARTED:
                # update params
                pass
            
            # if dash_stim_recall_resp is stopping this frame...
            if dash_stim_recall_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_recall_resp.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_recall_resp.tStop = t  # not accounting for scr refresh
                    dash_stim_recall_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_recall_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall_resp.stopped')
                    # update status
                    dash_stim_recall_resp.status = FINISHED
                    dash_stim_recall_resp.setAutoDraw(False)
            
            # *cue_stim_recall_resp* updates
            
            # if cue_stim_recall_resp is starting this frame...
            if cue_stim_recall_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_recall_resp.frameNStart = frameN  # exact frame index
                cue_stim_recall_resp.tStart = t  # local t and not account for scr refresh
                cue_stim_recall_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_recall_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_recall_resp.started')
                # update status
                cue_stim_recall_resp.status = STARTED
                cue_stim_recall_resp.setAutoDraw(True)
            
            # if cue_stim_recall_resp is active this frame...
            if cue_stim_recall_resp.status == STARTED:
                # update params
                pass
            
            # if cue_stim_recall_resp is stopping this frame...
            if cue_stim_recall_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_recall_resp.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_recall_resp.tStop = t  # not accounting for scr refresh
                    cue_stim_recall_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_recall_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall_resp.stopped')
                    # update status
                    cue_stim_recall_resp.status = FINISHED
                    cue_stim_recall_resp.setAutoDraw(False)
            
            # *recall_stim_resp* updates
            
            # if recall_stim_resp is starting this frame...
            if recall_stim_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_stim_resp.frameNStart = frameN  # exact frame index
                recall_stim_resp.tStart = t  # local t and not account for scr refresh
                recall_stim_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_stim_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_stim_resp.started')
                # update status
                recall_stim_resp.status = STARTED
                recall_stim_resp.setAutoDraw(True)
            
            # if recall_stim_resp is active this frame...
            if recall_stim_resp.status == STARTED:
                # update params
                pass
            
            # if recall_stim_resp is stopping this frame...
            if recall_stim_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_stim_resp.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_stim_resp.tStop = t  # not accounting for scr refresh
                    recall_stim_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_stim_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_stim_resp.stopped')
                    # update status
                    recall_stim_resp.status = FINISHED
                    recall_stim_resp.setAutoDraw(False)
            
            # *recall_question* updates
            
            # if recall_question is starting this frame...
            if recall_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_question.frameNStart = frameN  # exact frame index
                recall_question.tStart = t  # local t and not account for scr refresh
                recall_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_question.started')
                # update status
                recall_question.status = STARTED
                recall_question.setAutoDraw(True)
            
            # if recall_question is active this frame...
            if recall_question.status == STARTED:
                # update params
                pass
            
            # if recall_question is stopping this frame...
            if recall_question.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_question.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_question.tStop = t  # not accounting for scr refresh
                    recall_question.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_question.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_question.stopped')
                    # update status
                    recall_question.status = FINISHED
                    recall_question.setAutoDraw(False)
            
            # *recall_reached* updates
            waitOnFlip = False
            
            # if recall_reached is starting this frame...
            if recall_reached.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_reached.frameNStart = frameN  # exact frame index
                recall_reached.tStart = t  # local t and not account for scr refresh
                recall_reached.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_reached, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_reached.started')
                # update status
                recall_reached.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recall_reached.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recall_reached.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if recall_reached is stopping this frame...
            if recall_reached.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_reached.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_reached.tStop = t  # not accounting for scr refresh
                    recall_reached.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_reached.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_reached.stopped')
                    # update status
                    recall_reached.status = FINISHED
                    recall_reached.status = FINISHED
            if recall_reached.status == STARTED and not waitOnFlip:
                theseKeys = recall_reached.getKeys(keyList=['f', 'g'], ignoreKeys=["escape"], waitRelease=False)
                _recall_reached_allKeys.extend(theseKeys)
                if len(_recall_reached_allKeys):
                    recall_reached.keys = _recall_reached_allKeys[-1].name  # just the last key pressed
                    recall_reached.rt = _recall_reached_allKeys[-1].rt
                    recall_reached.duration = _recall_reached_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_response" ---
        for thisComponent in recall_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_response
        recall_response.tStop = globalClock.getTime(format='float')
        recall_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recall_response.stopped', recall_response.tStop)
        # check responses
        if recall_reached.keys in ['', [], None]:  # No response was made
            recall_reached.keys = None
        test_practice_trials.addData('recall_reached.keys',recall_reached.keys)
        if recall_reached.keys != None:  # we had a response
            test_practice_trials.addData('recall_reached.rt', recall_reached.rt)
            test_practice_trials.addData('recall_reached.duration', recall_reached.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if recall_response.maxDurationReached:
            routineTimer.addTime(-recall_response.maxDuration)
        elif recall_response.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "recall_select" ---
        # create an object to store info about Routine recall_select
        recall_select = data.Routine(
            name='recall_select',
            components=[dash_stim_recall_select, cue_stim_recall_select, recall_stim_select, recall_question_select, recall_selection],
        )
        recall_select.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from letter_choice
        recall_select_text = "Select the last letter of the word with a numeric key.\n"
        
        random_letters = rnd.choices("abcdefghijklmnoprstuvz", k=3)
        target_letter = target[-1]
        while target_letter in random_letters:
            random_letters = rnd.choices("abcdefghijklmnoprstuvz", k=3)
        
        
        random_letters.append(target_letter)
        print(random_letters)
        letters = rnd.sample(random_letters, len(random_letters))
        print(letters)
        recall_select_text += "1) " + letters[0] + "        "
        recall_select_text += "2) " + letters[1] + "        "
        recall_select_text += "3) " + letters[2] + "        "
        recall_select_text += "4) " + letters[3] 
        
        cue_stim_recall_select.setText(cue)
        recall_question_select.setText(recall_select_text)
        # create starting attributes for recall_selection
        recall_selection.keys = []
        recall_selection.rt = []
        _recall_selection_allKeys = []
        # store start times for recall_select
        recall_select.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_select.tStart = globalClock.getTime(format='float')
        recall_select.status = STARTED
        thisExp.addData('recall_select.started', recall_select.tStart)
        recall_select.maxDuration = None
        # skip Routine recall_select if its 'Skip if' condition is True
        recall_select.skipped = continueRoutine and not (recall_reached.keys != 'g')
        continueRoutine = recall_select.skipped
        # keep track of which components have finished
        recall_selectComponents = recall_select.components
        for thisComponent in recall_select.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_select" ---
        # if trial has changed, end Routine now
        if isinstance(test_practice_trials, data.TrialHandler2) and thisTest_practice_trial.thisN != test_practice_trials.thisTrial.thisN:
            continueRoutine = False
        recall_select.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_recall_select* updates
            
            # if dash_stim_recall_select is starting this frame...
            if dash_stim_recall_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_recall_select.frameNStart = frameN  # exact frame index
                dash_stim_recall_select.tStart = t  # local t and not account for scr refresh
                dash_stim_recall_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_recall_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_recall_select.started')
                # update status
                dash_stim_recall_select.status = STARTED
                dash_stim_recall_select.setAutoDraw(True)
            
            # if dash_stim_recall_select is active this frame...
            if dash_stim_recall_select.status == STARTED:
                # update params
                pass
            
            # if dash_stim_recall_select is stopping this frame...
            if dash_stim_recall_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_recall_select.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_recall_select.tStop = t  # not accounting for scr refresh
                    dash_stim_recall_select.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_recall_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall_select.stopped')
                    # update status
                    dash_stim_recall_select.status = FINISHED
                    dash_stim_recall_select.setAutoDraw(False)
            
            # *cue_stim_recall_select* updates
            
            # if cue_stim_recall_select is starting this frame...
            if cue_stim_recall_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_recall_select.frameNStart = frameN  # exact frame index
                cue_stim_recall_select.tStart = t  # local t and not account for scr refresh
                cue_stim_recall_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_recall_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_recall_select.started')
                # update status
                cue_stim_recall_select.status = STARTED
                cue_stim_recall_select.setAutoDraw(True)
            
            # if cue_stim_recall_select is active this frame...
            if cue_stim_recall_select.status == STARTED:
                # update params
                pass
            
            # if cue_stim_recall_select is stopping this frame...
            if cue_stim_recall_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_recall_select.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_recall_select.tStop = t  # not accounting for scr refresh
                    cue_stim_recall_select.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_recall_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall_select.stopped')
                    # update status
                    cue_stim_recall_select.status = FINISHED
                    cue_stim_recall_select.setAutoDraw(False)
            
            # *recall_stim_select* updates
            
            # if recall_stim_select is starting this frame...
            if recall_stim_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_stim_select.frameNStart = frameN  # exact frame index
                recall_stim_select.tStart = t  # local t and not account for scr refresh
                recall_stim_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_stim_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_stim_select.started')
                # update status
                recall_stim_select.status = STARTED
                recall_stim_select.setAutoDraw(True)
            
            # if recall_stim_select is active this frame...
            if recall_stim_select.status == STARTED:
                # update params
                pass
            
            # if recall_stim_select is stopping this frame...
            if recall_stim_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_stim_select.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_stim_select.tStop = t  # not accounting for scr refresh
                    recall_stim_select.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_stim_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_stim_select.stopped')
                    # update status
                    recall_stim_select.status = FINISHED
                    recall_stim_select.setAutoDraw(False)
            
            # *recall_question_select* updates
            
            # if recall_question_select is starting this frame...
            if recall_question_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_question_select.frameNStart = frameN  # exact frame index
                recall_question_select.tStart = t  # local t and not account for scr refresh
                recall_question_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_question_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_question_select.started')
                # update status
                recall_question_select.status = STARTED
                recall_question_select.setAutoDraw(True)
            
            # if recall_question_select is active this frame...
            if recall_question_select.status == STARTED:
                # update params
                pass
            
            # if recall_question_select is stopping this frame...
            if recall_question_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_question_select.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_question_select.tStop = t  # not accounting for scr refresh
                    recall_question_select.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_question_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_question_select.stopped')
                    # update status
                    recall_question_select.status = FINISHED
                    recall_question_select.setAutoDraw(False)
            
            # *recall_selection* updates
            waitOnFlip = False
            
            # if recall_selection is starting this frame...
            if recall_selection.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_selection.frameNStart = frameN  # exact frame index
                recall_selection.tStart = t  # local t and not account for scr refresh
                recall_selection.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_selection, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_selection.started')
                # update status
                recall_selection.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recall_selection.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recall_selection.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if recall_selection is stopping this frame...
            if recall_selection.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_selection.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_selection.tStop = t  # not accounting for scr refresh
                    recall_selection.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_selection.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_selection.stopped')
                    # update status
                    recall_selection.status = FINISHED
                    recall_selection.status = FINISHED
            if recall_selection.status == STARTED and not waitOnFlip:
                theseKeys = recall_selection.getKeys(keyList=['1', '2', '3', '4'], ignoreKeys=["escape"], waitRelease=False)
                _recall_selection_allKeys.extend(theseKeys)
                if len(_recall_selection_allKeys):
                    recall_selection.keys = _recall_selection_allKeys[-1].name  # just the last key pressed
                    recall_selection.rt = _recall_selection_allKeys[-1].rt
                    recall_selection.duration = _recall_selection_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_select.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_select.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_select" ---
        for thisComponent in recall_select.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_select
        recall_select.tStop = globalClock.getTime(format='float')
        recall_select.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recall_select.stopped', recall_select.tStop)
        # check responses
        if recall_selection.keys in ['', [], None]:  # No response was made
            recall_selection.keys = None
        test_practice_trials.addData('recall_selection.keys',recall_selection.keys)
        if recall_selection.keys != None:  # we had a response
            test_practice_trials.addData('recall_selection.rt', recall_selection.rt)
            test_practice_trials.addData('recall_selection.duration', recall_selection.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if recall_select.maxDurationReached:
            routineTimer.addTime(-recall_select.maxDuration)
        elif recall_select.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'test_practice_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "start_task" ---
    # create an object to store info about Routine start_task
    start_task = data.Routine(
        name='start_task',
        components=[start_task_text, start_key],
    )
    start_task.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for start_key
    start_key.keys = []
    start_key.rt = []
    _start_key_allKeys = []
    # store start times for start_task
    start_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start_task.tStart = globalClock.getTime(format='float')
    start_task.status = STARTED
    thisExp.addData('start_task.started', start_task.tStart)
    start_task.maxDuration = None
    # keep track of which components have finished
    start_taskComponents = start_task.components
    for thisComponent in start_task.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_task" ---
    start_task.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *start_task_text* updates
        
        # if start_task_text is starting this frame...
        if start_task_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_task_text.frameNStart = frameN  # exact frame index
            start_task_text.tStart = t  # local t and not account for scr refresh
            start_task_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_task_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_task_text.started')
            # update status
            start_task_text.status = STARTED
            start_task_text.setAutoDraw(True)
        
        # if start_task_text is active this frame...
        if start_task_text.status == STARTED:
            # update params
            pass
        
        # *start_key* updates
        waitOnFlip = False
        
        # if start_key is starting this frame...
        if start_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            start_key.frameNStart = frameN  # exact frame index
            start_key.tStart = t  # local t and not account for scr refresh
            start_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(start_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'start_key.started')
            # update status
            start_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(start_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(start_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if start_key.status == STARTED and not waitOnFlip:
            theseKeys = start_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _start_key_allKeys.extend(theseKeys)
            if len(_start_key_allKeys):
                start_key.keys = _start_key_allKeys[-1].name  # just the last key pressed
                start_key.rt = _start_key_allKeys[-1].rt
                start_key.duration = _start_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start_task.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_task.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_task" ---
    for thisComponent in start_task.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start_task
    start_task.tStop = globalClock.getTime(format='float')
    start_task.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start_task.stopped', start_task.tStop)
    # check responses
    if start_key.keys in ['', [], None]:  # No response was made
        start_key.keys = None
    thisExp.addData('start_key.keys',start_key.keys)
    if start_key.keys != None:  # we had a response
        thisExp.addData('start_key.rt', start_key.rt)
        thisExp.addData('start_key.duration', start_key.duration)
    thisExp.nextEntry()
    # the Routine "start_task" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    test_trials = data.TrialHandler2(
        name='test_trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('cue_target_pairs.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(test_trials)  # add the loop to the experiment
    thisTest_trial = test_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTest_trial.rgb)
    if thisTest_trial != None:
        for paramName in thisTest_trial:
            globals()[paramName] = thisTest_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTest_trial in test_trials:
        currentLoop = test_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTest_trial.rgb)
        if thisTest_trial != None:
            for paramName in thisTest_trial:
                globals()[paramName] = thisTest_trial[paramName]
        
        # --- Prepare to start Routine "iti_recall" ---
        # create an object to store info about Routine iti_recall
        iti_recall = data.Routine(
            name='iti_recall',
            components=[ISI_2],
        )
        iti_recall.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for iti_recall
        iti_recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iti_recall.tStart = globalClock.getTime(format='float')
        iti_recall.status = STARTED
        thisExp.addData('iti_recall.started', iti_recall.tStart)
        iti_recall.maxDuration = None
        # keep track of which components have finished
        iti_recallComponents = iti_recall.components
        for thisComponent in iti_recall.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iti_recall" ---
        # if trial has changed, end Routine now
        if isinstance(test_trials, data.TrialHandler2) and thisTest_trial.thisN != test_trials.thisTrial.thisN:
            continueRoutine = False
        iti_recall.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *ISI_2* period
            
            # if ISI_2 is starting this frame...
            if ISI_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ISI_2.frameNStart = frameN  # exact frame index
                ISI_2.tStart = t  # local t and not account for scr refresh
                ISI_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ISI_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('ISI_2.started', t)
                # update status
                ISI_2.status = STARTED
                ISI_2.start(2.0)
            elif ISI_2.status == STARTED:  # one frame should pass before updating params and completing
                ISI_2.complete()  # finish the static period
                ISI_2.tStop = ISI_2.tStart + 2.0  # record stop time
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                iti_recall.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in iti_recall.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti_recall" ---
        for thisComponent in iti_recall.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iti_recall
        iti_recall.tStop = globalClock.getTime(format='float')
        iti_recall.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iti_recall.stopped', iti_recall.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if iti_recall.maxDurationReached:
            routineTimer.addTime(-iti_recall.maxDuration)
        elif iti_recall.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "cued_recall" ---
        # create an object to store info about Routine cued_recall
        cued_recall = data.Routine(
            name='cued_recall',
            components=[cue_stim_recall, dash_stim_recall, qmark_target, end_cued_recall],
        )
        cued_recall.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_recall.setText(cue)
        # create starting attributes for end_cued_recall
        end_cued_recall.keys = []
        end_cued_recall.rt = []
        _end_cued_recall_allKeys = []
        # store start times for cued_recall
        cued_recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        cued_recall.tStart = globalClock.getTime(format='float')
        cued_recall.status = STARTED
        thisExp.addData('cued_recall.started', cued_recall.tStart)
        cued_recall.maxDuration = None
        # keep track of which components have finished
        cued_recallComponents = cued_recall.components
        for thisComponent in cued_recall.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "cued_recall" ---
        # if trial has changed, end Routine now
        if isinstance(test_trials, data.TrialHandler2) and thisTest_trial.thisN != test_trials.thisTrial.thisN:
            continueRoutine = False
        cued_recall.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cue_stim_recall* updates
            
            # if cue_stim_recall is starting this frame...
            if cue_stim_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_recall.frameNStart = frameN  # exact frame index
                cue_stim_recall.tStart = t  # local t and not account for scr refresh
                cue_stim_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_recall, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_recall.started')
                # update status
                cue_stim_recall.status = STARTED
                cue_stim_recall.setAutoDraw(True)
            
            # if cue_stim_recall is active this frame...
            if cue_stim_recall.status == STARTED:
                # update params
                pass
            
            # if cue_stim_recall is stopping this frame...
            if cue_stim_recall.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_recall.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_recall.tStop = t  # not accounting for scr refresh
                    cue_stim_recall.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_recall.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall.stopped')
                    # update status
                    cue_stim_recall.status = FINISHED
                    cue_stim_recall.setAutoDraw(False)
            
            # *dash_stim_recall* updates
            
            # if dash_stim_recall is starting this frame...
            if dash_stim_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_recall.frameNStart = frameN  # exact frame index
                dash_stim_recall.tStart = t  # local t and not account for scr refresh
                dash_stim_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_recall, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_recall.started')
                # update status
                dash_stim_recall.status = STARTED
                dash_stim_recall.setAutoDraw(True)
            
            # if dash_stim_recall is active this frame...
            if dash_stim_recall.status == STARTED:
                # update params
                pass
            
            # if dash_stim_recall is stopping this frame...
            if dash_stim_recall.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_recall.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_recall.tStop = t  # not accounting for scr refresh
                    dash_stim_recall.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_recall.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall.stopped')
                    # update status
                    dash_stim_recall.status = FINISHED
                    dash_stim_recall.setAutoDraw(False)
            
            # *qmark_target* updates
            
            # if qmark_target is starting this frame...
            if qmark_target.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                qmark_target.frameNStart = frameN  # exact frame index
                qmark_target.tStart = t  # local t and not account for scr refresh
                qmark_target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(qmark_target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'qmark_target.started')
                # update status
                qmark_target.status = STARTED
                qmark_target.setAutoDraw(True)
            
            # if qmark_target is active this frame...
            if qmark_target.status == STARTED:
                # update params
                pass
            
            # if qmark_target is stopping this frame...
            if qmark_target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > qmark_target.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    qmark_target.tStop = t  # not accounting for scr refresh
                    qmark_target.tStopRefresh = tThisFlipGlobal  # on global time
                    qmark_target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'qmark_target.stopped')
                    # update status
                    qmark_target.status = FINISHED
                    qmark_target.setAutoDraw(False)
            
            # *end_cued_recall* updates
            waitOnFlip = False
            
            # if end_cued_recall is starting this frame...
            if end_cued_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_cued_recall.frameNStart = frameN  # exact frame index
                end_cued_recall.tStart = t  # local t and not account for scr refresh
                end_cued_recall.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_cued_recall, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_cued_recall.started')
                # update status
                end_cued_recall.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_cued_recall.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_cued_recall.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if end_cued_recall is stopping this frame...
            if end_cued_recall.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > end_cued_recall.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    end_cued_recall.tStop = t  # not accounting for scr refresh
                    end_cued_recall.tStopRefresh = tThisFlipGlobal  # on global time
                    end_cued_recall.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_cued_recall.stopped')
                    # update status
                    end_cued_recall.status = FINISHED
                    end_cued_recall.status = FINISHED
            if end_cued_recall.status == STARTED and not waitOnFlip:
                theseKeys = end_cued_recall.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _end_cued_recall_allKeys.extend(theseKeys)
                if len(_end_cued_recall_allKeys):
                    end_cued_recall.keys = _end_cued_recall_allKeys[-1].name  # just the last key pressed
                    end_cued_recall.rt = _end_cued_recall_allKeys[-1].rt
                    end_cued_recall.duration = _end_cued_recall_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                cued_recall.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in cued_recall.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "cued_recall" ---
        for thisComponent in cued_recall.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for cued_recall
        cued_recall.tStop = globalClock.getTime(format='float')
        cued_recall.tStopRefresh = tThisFlipGlobal
        thisExp.addData('cued_recall.stopped', cued_recall.tStop)
        # check responses
        if end_cued_recall.keys in ['', [], None]:  # No response was made
            end_cued_recall.keys = None
        test_trials.addData('end_cued_recall.keys',end_cued_recall.keys)
        if end_cued_recall.keys != None:  # we had a response
            test_trials.addData('end_cued_recall.rt', end_cued_recall.rt)
            test_trials.addData('end_cued_recall.duration', end_cued_recall.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if cued_recall.maxDurationReached:
            routineTimer.addTime(-cued_recall.maxDuration)
        elif cued_recall.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        
        # --- Prepare to start Routine "recall_response" ---
        # create an object to store info about Routine recall_response
        recall_response = data.Routine(
            name='recall_response',
            components=[dash_stim_recall_resp, cue_stim_recall_resp, recall_stim_resp, recall_question, recall_reached],
        )
        recall_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        cue_stim_recall_resp.setText(cue)
        # create starting attributes for recall_reached
        recall_reached.keys = []
        recall_reached.rt = []
        _recall_reached_allKeys = []
        # store start times for recall_response
        recall_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_response.tStart = globalClock.getTime(format='float')
        recall_response.status = STARTED
        thisExp.addData('recall_response.started', recall_response.tStart)
        recall_response.maxDuration = None
        # keep track of which components have finished
        recall_responseComponents = recall_response.components
        for thisComponent in recall_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_response" ---
        # if trial has changed, end Routine now
        if isinstance(test_trials, data.TrialHandler2) and thisTest_trial.thisN != test_trials.thisTrial.thisN:
            continueRoutine = False
        recall_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_recall_resp* updates
            
            # if dash_stim_recall_resp is starting this frame...
            if dash_stim_recall_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_recall_resp.frameNStart = frameN  # exact frame index
                dash_stim_recall_resp.tStart = t  # local t and not account for scr refresh
                dash_stim_recall_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_recall_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_recall_resp.started')
                # update status
                dash_stim_recall_resp.status = STARTED
                dash_stim_recall_resp.setAutoDraw(True)
            
            # if dash_stim_recall_resp is active this frame...
            if dash_stim_recall_resp.status == STARTED:
                # update params
                pass
            
            # if dash_stim_recall_resp is stopping this frame...
            if dash_stim_recall_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_recall_resp.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_recall_resp.tStop = t  # not accounting for scr refresh
                    dash_stim_recall_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_recall_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall_resp.stopped')
                    # update status
                    dash_stim_recall_resp.status = FINISHED
                    dash_stim_recall_resp.setAutoDraw(False)
            
            # *cue_stim_recall_resp* updates
            
            # if cue_stim_recall_resp is starting this frame...
            if cue_stim_recall_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_recall_resp.frameNStart = frameN  # exact frame index
                cue_stim_recall_resp.tStart = t  # local t and not account for scr refresh
                cue_stim_recall_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_recall_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_recall_resp.started')
                # update status
                cue_stim_recall_resp.status = STARTED
                cue_stim_recall_resp.setAutoDraw(True)
            
            # if cue_stim_recall_resp is active this frame...
            if cue_stim_recall_resp.status == STARTED:
                # update params
                pass
            
            # if cue_stim_recall_resp is stopping this frame...
            if cue_stim_recall_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_recall_resp.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_recall_resp.tStop = t  # not accounting for scr refresh
                    cue_stim_recall_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_recall_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall_resp.stopped')
                    # update status
                    cue_stim_recall_resp.status = FINISHED
                    cue_stim_recall_resp.setAutoDraw(False)
            
            # *recall_stim_resp* updates
            
            # if recall_stim_resp is starting this frame...
            if recall_stim_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_stim_resp.frameNStart = frameN  # exact frame index
                recall_stim_resp.tStart = t  # local t and not account for scr refresh
                recall_stim_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_stim_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_stim_resp.started')
                # update status
                recall_stim_resp.status = STARTED
                recall_stim_resp.setAutoDraw(True)
            
            # if recall_stim_resp is active this frame...
            if recall_stim_resp.status == STARTED:
                # update params
                pass
            
            # if recall_stim_resp is stopping this frame...
            if recall_stim_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_stim_resp.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_stim_resp.tStop = t  # not accounting for scr refresh
                    recall_stim_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_stim_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_stim_resp.stopped')
                    # update status
                    recall_stim_resp.status = FINISHED
                    recall_stim_resp.setAutoDraw(False)
            
            # *recall_question* updates
            
            # if recall_question is starting this frame...
            if recall_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_question.frameNStart = frameN  # exact frame index
                recall_question.tStart = t  # local t and not account for scr refresh
                recall_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_question.started')
                # update status
                recall_question.status = STARTED
                recall_question.setAutoDraw(True)
            
            # if recall_question is active this frame...
            if recall_question.status == STARTED:
                # update params
                pass
            
            # if recall_question is stopping this frame...
            if recall_question.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_question.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_question.tStop = t  # not accounting for scr refresh
                    recall_question.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_question.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_question.stopped')
                    # update status
                    recall_question.status = FINISHED
                    recall_question.setAutoDraw(False)
            
            # *recall_reached* updates
            waitOnFlip = False
            
            # if recall_reached is starting this frame...
            if recall_reached.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_reached.frameNStart = frameN  # exact frame index
                recall_reached.tStart = t  # local t and not account for scr refresh
                recall_reached.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_reached, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_reached.started')
                # update status
                recall_reached.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recall_reached.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recall_reached.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if recall_reached is stopping this frame...
            if recall_reached.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_reached.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_reached.tStop = t  # not accounting for scr refresh
                    recall_reached.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_reached.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_reached.stopped')
                    # update status
                    recall_reached.status = FINISHED
                    recall_reached.status = FINISHED
            if recall_reached.status == STARTED and not waitOnFlip:
                theseKeys = recall_reached.getKeys(keyList=['f', 'g'], ignoreKeys=["escape"], waitRelease=False)
                _recall_reached_allKeys.extend(theseKeys)
                if len(_recall_reached_allKeys):
                    recall_reached.keys = _recall_reached_allKeys[-1].name  # just the last key pressed
                    recall_reached.rt = _recall_reached_allKeys[-1].rt
                    recall_reached.duration = _recall_reached_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_response" ---
        for thisComponent in recall_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_response
        recall_response.tStop = globalClock.getTime(format='float')
        recall_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recall_response.stopped', recall_response.tStop)
        # check responses
        if recall_reached.keys in ['', [], None]:  # No response was made
            recall_reached.keys = None
        test_trials.addData('recall_reached.keys',recall_reached.keys)
        if recall_reached.keys != None:  # we had a response
            test_trials.addData('recall_reached.rt', recall_reached.rt)
            test_trials.addData('recall_reached.duration', recall_reached.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if recall_response.maxDurationReached:
            routineTimer.addTime(-recall_response.maxDuration)
        elif recall_response.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "recall_select" ---
        # create an object to store info about Routine recall_select
        recall_select = data.Routine(
            name='recall_select',
            components=[dash_stim_recall_select, cue_stim_recall_select, recall_stim_select, recall_question_select, recall_selection],
        )
        recall_select.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from letter_choice
        recall_select_text = "Select the last letter of the word with a numeric key.\n"
        
        random_letters = rnd.choices("abcdefghijklmnoprstuvz", k=3)
        target_letter = target[-1]
        while target_letter in random_letters:
            random_letters = rnd.choices("abcdefghijklmnoprstuvz", k=3)
        
        
        random_letters.append(target_letter)
        print(random_letters)
        letters = rnd.sample(random_letters, len(random_letters))
        print(letters)
        recall_select_text += "1) " + letters[0] + "        "
        recall_select_text += "2) " + letters[1] + "        "
        recall_select_text += "3) " + letters[2] + "        "
        recall_select_text += "4) " + letters[3] 
        
        cue_stim_recall_select.setText(cue)
        recall_question_select.setText(recall_select_text)
        # create starting attributes for recall_selection
        recall_selection.keys = []
        recall_selection.rt = []
        _recall_selection_allKeys = []
        # store start times for recall_select
        recall_select.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        recall_select.tStart = globalClock.getTime(format='float')
        recall_select.status = STARTED
        thisExp.addData('recall_select.started', recall_select.tStart)
        recall_select.maxDuration = None
        # skip Routine recall_select if its 'Skip if' condition is True
        recall_select.skipped = continueRoutine and not (recall_reached.keys != 'g')
        continueRoutine = recall_select.skipped
        # keep track of which components have finished
        recall_selectComponents = recall_select.components
        for thisComponent in recall_select.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "recall_select" ---
        # if trial has changed, end Routine now
        if isinstance(test_trials, data.TrialHandler2) and thisTest_trial.thisN != test_trials.thisTrial.thisN:
            continueRoutine = False
        recall_select.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *dash_stim_recall_select* updates
            
            # if dash_stim_recall_select is starting this frame...
            if dash_stim_recall_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                dash_stim_recall_select.frameNStart = frameN  # exact frame index
                dash_stim_recall_select.tStart = t  # local t and not account for scr refresh
                dash_stim_recall_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(dash_stim_recall_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'dash_stim_recall_select.started')
                # update status
                dash_stim_recall_select.status = STARTED
                dash_stim_recall_select.setAutoDraw(True)
            
            # if dash_stim_recall_select is active this frame...
            if dash_stim_recall_select.status == STARTED:
                # update params
                pass
            
            # if dash_stim_recall_select is stopping this frame...
            if dash_stim_recall_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > dash_stim_recall_select.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    dash_stim_recall_select.tStop = t  # not accounting for scr refresh
                    dash_stim_recall_select.tStopRefresh = tThisFlipGlobal  # on global time
                    dash_stim_recall_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'dash_stim_recall_select.stopped')
                    # update status
                    dash_stim_recall_select.status = FINISHED
                    dash_stim_recall_select.setAutoDraw(False)
            
            # *cue_stim_recall_select* updates
            
            # if cue_stim_recall_select is starting this frame...
            if cue_stim_recall_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim_recall_select.frameNStart = frameN  # exact frame index
                cue_stim_recall_select.tStart = t  # local t and not account for scr refresh
                cue_stim_recall_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim_recall_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim_recall_select.started')
                # update status
                cue_stim_recall_select.status = STARTED
                cue_stim_recall_select.setAutoDraw(True)
            
            # if cue_stim_recall_select is active this frame...
            if cue_stim_recall_select.status == STARTED:
                # update params
                pass
            
            # if cue_stim_recall_select is stopping this frame...
            if cue_stim_recall_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim_recall_select.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim_recall_select.tStop = t  # not accounting for scr refresh
                    cue_stim_recall_select.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim_recall_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim_recall_select.stopped')
                    # update status
                    cue_stim_recall_select.status = FINISHED
                    cue_stim_recall_select.setAutoDraw(False)
            
            # *recall_stim_select* updates
            
            # if recall_stim_select is starting this frame...
            if recall_stim_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_stim_select.frameNStart = frameN  # exact frame index
                recall_stim_select.tStart = t  # local t and not account for scr refresh
                recall_stim_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_stim_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_stim_select.started')
                # update status
                recall_stim_select.status = STARTED
                recall_stim_select.setAutoDraw(True)
            
            # if recall_stim_select is active this frame...
            if recall_stim_select.status == STARTED:
                # update params
                pass
            
            # if recall_stim_select is stopping this frame...
            if recall_stim_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_stim_select.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_stim_select.tStop = t  # not accounting for scr refresh
                    recall_stim_select.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_stim_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_stim_select.stopped')
                    # update status
                    recall_stim_select.status = FINISHED
                    recall_stim_select.setAutoDraw(False)
            
            # *recall_question_select* updates
            
            # if recall_question_select is starting this frame...
            if recall_question_select.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_question_select.frameNStart = frameN  # exact frame index
                recall_question_select.tStart = t  # local t and not account for scr refresh
                recall_question_select.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_question_select, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_question_select.started')
                # update status
                recall_question_select.status = STARTED
                recall_question_select.setAutoDraw(True)
            
            # if recall_question_select is active this frame...
            if recall_question_select.status == STARTED:
                # update params
                pass
            
            # if recall_question_select is stopping this frame...
            if recall_question_select.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_question_select.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_question_select.tStop = t  # not accounting for scr refresh
                    recall_question_select.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_question_select.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_question_select.stopped')
                    # update status
                    recall_question_select.status = FINISHED
                    recall_question_select.setAutoDraw(False)
            
            # *recall_selection* updates
            waitOnFlip = False
            
            # if recall_selection is starting this frame...
            if recall_selection.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                recall_selection.frameNStart = frameN  # exact frame index
                recall_selection.tStart = t  # local t and not account for scr refresh
                recall_selection.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(recall_selection, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_selection.started')
                # update status
                recall_selection.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(recall_selection.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(recall_selection.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if recall_selection is stopping this frame...
            if recall_selection.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > recall_selection.tStartRefresh + 4.0-frameTolerance:
                    # keep track of stop time/frame for later
                    recall_selection.tStop = t  # not accounting for scr refresh
                    recall_selection.tStopRefresh = tThisFlipGlobal  # on global time
                    recall_selection.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'recall_selection.stopped')
                    # update status
                    recall_selection.status = FINISHED
                    recall_selection.status = FINISHED
            if recall_selection.status == STARTED and not waitOnFlip:
                theseKeys = recall_selection.getKeys(keyList=['1', '2', '3', '4'], ignoreKeys=["escape"], waitRelease=False)
                _recall_selection_allKeys.extend(theseKeys)
                if len(_recall_selection_allKeys):
                    recall_selection.keys = _recall_selection_allKeys[-1].name  # just the last key pressed
                    recall_selection.rt = _recall_selection_allKeys[-1].rt
                    recall_selection.duration = _recall_selection_allKeys[-1].duration
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                recall_select.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in recall_select.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "recall_select" ---
        for thisComponent in recall_select.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for recall_select
        recall_select.tStop = globalClock.getTime(format='float')
        recall_select.tStopRefresh = tThisFlipGlobal
        thisExp.addData('recall_select.stopped', recall_select.tStop)
        # check responses
        if recall_selection.keys in ['', [], None]:  # No response was made
            recall_selection.keys = None
        test_trials.addData('recall_selection.keys',recall_selection.keys)
        if recall_selection.keys != None:  # we had a response
            test_trials.addData('recall_selection.rt', recall_selection.rt)
            test_trials.addData('recall_selection.duration', recall_selection.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if recall_select.maxDurationReached:
            routineTimer.addTime(-recall_select.maxDuration)
        elif recall_select.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'test_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end_part2" ---
    # create an object to store info about Routine end_part2
    end_part2 = data.Routine(
        name='end_part2',
        components=[end_part2_key, end_part2_text],
    )
    end_part2.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for end_part2_key
    end_part2_key.keys = []
    end_part2_key.rt = []
    _end_part2_key_allKeys = []
    # store start times for end_part2
    end_part2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_part2.tStart = globalClock.getTime(format='float')
    end_part2.status = STARTED
    thisExp.addData('end_part2.started', end_part2.tStart)
    end_part2.maxDuration = None
    # keep track of which components have finished
    end_part2Components = end_part2.components
    for thisComponent in end_part2.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end_part2" ---
    end_part2.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_part2_key* updates
        waitOnFlip = False
        
        # if end_part2_key is starting this frame...
        if end_part2_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_part2_key.frameNStart = frameN  # exact frame index
            end_part2_key.tStart = t  # local t and not account for scr refresh
            end_part2_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_part2_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_part2_key.started')
            # update status
            end_part2_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_part2_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_part2_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_part2_key.status == STARTED and not waitOnFlip:
            theseKeys = end_part2_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _end_part2_key_allKeys.extend(theseKeys)
            if len(_end_part2_key_allKeys):
                end_part2_key.keys = _end_part2_key_allKeys[-1].name  # just the last key pressed
                end_part2_key.rt = _end_part2_key_allKeys[-1].rt
                end_part2_key.duration = _end_part2_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *end_part2_text* updates
        
        # if end_part2_text is starting this frame...
        if end_part2_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_part2_text.frameNStart = frameN  # exact frame index
            end_part2_text.tStart = t  # local t and not account for scr refresh
            end_part2_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_part2_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_part2_text.started')
            # update status
            end_part2_text.status = STARTED
            end_part2_text.setAutoDraw(True)
        
        # if end_part2_text is active this frame...
        if end_part2_text.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end_part2.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_part2.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_part2" ---
    for thisComponent in end_part2.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_part2
    end_part2.tStop = globalClock.getTime(format='float')
    end_part2.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_part2.stopped', end_part2.tStop)
    # check responses
    if end_part2_key.keys in ['', [], None]:  # No response was made
        end_part2_key.keys = None
    thisExp.addData('end_part2_key.keys',end_part2_key.keys)
    if end_part2_key.keys != None:  # we had a response
        thisExp.addData('end_part2_key.rt', end_part2_key.rt)
        thisExp.addData('end_part2_key.duration', end_part2_key.duration)
    thisExp.nextEntry()
    # the Routine "end_part2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructions_semantic_mapping" ---
    # create an object to store info about Routine instructions_semantic_mapping
    instructions_semantic_mapping = data.Routine(
        name='instructions_semantic_mapping',
        components=[sm_instructions_text, sm_instructions_key],
    )
    instructions_semantic_mapping.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for sm_instructions_key
    sm_instructions_key.keys = []
    sm_instructions_key.rt = []
    _sm_instructions_key_allKeys = []
    # store start times for instructions_semantic_mapping
    instructions_semantic_mapping.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_semantic_mapping.tStart = globalClock.getTime(format='float')
    instructions_semantic_mapping.status = STARTED
    thisExp.addData('instructions_semantic_mapping.started', instructions_semantic_mapping.tStart)
    instructions_semantic_mapping.maxDuration = None
    # keep track of which components have finished
    instructions_semantic_mappingComponents = instructions_semantic_mapping.components
    for thisComponent in instructions_semantic_mapping.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_semantic_mapping" ---
    instructions_semantic_mapping.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *sm_instructions_text* updates
        
        # if sm_instructions_text is starting this frame...
        if sm_instructions_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sm_instructions_text.frameNStart = frameN  # exact frame index
            sm_instructions_text.tStart = t  # local t and not account for scr refresh
            sm_instructions_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sm_instructions_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'sm_instructions_text.started')
            # update status
            sm_instructions_text.status = STARTED
            sm_instructions_text.setAutoDraw(True)
        
        # if sm_instructions_text is active this frame...
        if sm_instructions_text.status == STARTED:
            # update params
            pass
        
        # *sm_instructions_key* updates
        waitOnFlip = False
        
        # if sm_instructions_key is starting this frame...
        if sm_instructions_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            sm_instructions_key.frameNStart = frameN  # exact frame index
            sm_instructions_key.tStart = t  # local t and not account for scr refresh
            sm_instructions_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(sm_instructions_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'sm_instructions_key.started')
            # update status
            sm_instructions_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(sm_instructions_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(sm_instructions_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if sm_instructions_key.status == STARTED and not waitOnFlip:
            theseKeys = sm_instructions_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _sm_instructions_key_allKeys.extend(theseKeys)
            if len(_sm_instructions_key_allKeys):
                sm_instructions_key.keys = _sm_instructions_key_allKeys[-1].name  # just the last key pressed
                sm_instructions_key.rt = _sm_instructions_key_allKeys[-1].rt
                sm_instructions_key.duration = _sm_instructions_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions_semantic_mapping.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions_semantic_mapping.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_semantic_mapping" ---
    for thisComponent in instructions_semantic_mapping.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_semantic_mapping
    instructions_semantic_mapping.tStop = globalClock.getTime(format='float')
    instructions_semantic_mapping.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_semantic_mapping.stopped', instructions_semantic_mapping.tStop)
    # check responses
    if sm_instructions_key.keys in ['', [], None]:  # No response was made
        sm_instructions_key.keys = None
    thisExp.addData('sm_instructions_key.keys',sm_instructions_key.keys)
    if sm_instructions_key.keys != None:  # we had a response
        thisExp.addData('sm_instructions_key.rt', sm_instructions_key.rt)
        thisExp.addData('sm_instructions_key.duration', sm_instructions_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions_semantic_mapping" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    semantic_mapping = data.TrialHandler2(
        name='semantic_mapping',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('targets_mediators.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(semantic_mapping)  # add the loop to the experiment
    thisSemantic_mapping = semantic_mapping.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSemantic_mapping.rgb)
    if thisSemantic_mapping != None:
        for paramName in thisSemantic_mapping:
            globals()[paramName] = thisSemantic_mapping[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisSemantic_mapping in semantic_mapping:
        currentLoop = semantic_mapping
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisSemantic_mapping.rgb)
        if thisSemantic_mapping != None:
            for paramName in thisSemantic_mapping:
                globals()[paramName] = thisSemantic_mapping[paramName]
        
        # --- Prepare to start Routine "iti_mapping" ---
        # create an object to store info about Routine iti_mapping
        iti_mapping = data.Routine(
            name='iti_mapping',
            components=[ISI_3],
        )
        iti_mapping.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from mapping_trial_counter
        mapping_trial = mapping_trial + 1
        # store start times for iti_mapping
        iti_mapping.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iti_mapping.tStart = globalClock.getTime(format='float')
        iti_mapping.status = STARTED
        thisExp.addData('iti_mapping.started', iti_mapping.tStart)
        iti_mapping.maxDuration = None
        # keep track of which components have finished
        iti_mappingComponents = iti_mapping.components
        for thisComponent in iti_mapping.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iti_mapping" ---
        # if trial has changed, end Routine now
        if isinstance(semantic_mapping, data.TrialHandler2) and thisSemantic_mapping.thisN != semantic_mapping.thisTrial.thisN:
            continueRoutine = False
        iti_mapping.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # *ISI_3* period
            
            # if ISI_3 is starting this frame...
            if ISI_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ISI_3.frameNStart = frameN  # exact frame index
                ISI_3.tStart = t  # local t and not account for scr refresh
                ISI_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ISI_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('ISI_3.started', t)
                # update status
                ISI_3.status = STARTED
                ISI_3.start(2.0)
            elif ISI_3.status == STARTED:  # one frame should pass before updating params and completing
                ISI_3.complete()  # finish the static period
                ISI_3.tStop = ISI_3.tStart + 2.0  # record stop time
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                iti_mapping.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in iti_mapping.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti_mapping" ---
        for thisComponent in iti_mapping.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iti_mapping
        iti_mapping.tStop = globalClock.getTime(format='float')
        iti_mapping.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iti_mapping.stopped', iti_mapping.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if iti_mapping.maxDurationReached:
            routineTimer.addTime(-iti_mapping.maxDuration)
        elif iti_mapping.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "item" ---
        # create an object to store info about Routine item
        item = data.Routine(
            name='item',
            components=[semantic_map_item, living_nonliving, end_mapping, licing_nonliving_text],
        )
        item.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        semantic_map_item.setText(list_item)
        # create starting attributes for living_nonliving
        living_nonliving.keys = []
        living_nonliving.rt = []
        _living_nonliving_allKeys = []
        # create starting attributes for end_mapping
        end_mapping.keys = []
        end_mapping.rt = []
        _end_mapping_allKeys = []
        # store start times for item
        item.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        item.tStart = globalClock.getTime(format='float')
        item.status = STARTED
        thisExp.addData('item.started', item.tStart)
        item.maxDuration = None
        # keep track of which components have finished
        itemComponents = item.components
        for thisComponent in item.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "item" ---
        # if trial has changed, end Routine now
        if isinstance(semantic_mapping, data.TrialHandler2) and thisSemantic_mapping.thisN != semantic_mapping.thisTrial.thisN:
            continueRoutine = False
        item.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 5.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *semantic_map_item* updates
            
            # if semantic_map_item is starting this frame...
            if semantic_map_item.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                semantic_map_item.frameNStart = frameN  # exact frame index
                semantic_map_item.tStart = t  # local t and not account for scr refresh
                semantic_map_item.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(semantic_map_item, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'semantic_map_item.started')
                # update status
                semantic_map_item.status = STARTED
                semantic_map_item.setAutoDraw(True)
            
            # if semantic_map_item is active this frame...
            if semantic_map_item.status == STARTED:
                # update params
                pass
            
            # if semantic_map_item is stopping this frame...
            if semantic_map_item.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > semantic_map_item.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    semantic_map_item.tStop = t  # not accounting for scr refresh
                    semantic_map_item.tStopRefresh = tThisFlipGlobal  # on global time
                    semantic_map_item.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'semantic_map_item.stopped')
                    # update status
                    semantic_map_item.status = FINISHED
                    semantic_map_item.setAutoDraw(False)
            
            # *living_nonliving* updates
            waitOnFlip = False
            
            # if living_nonliving is starting this frame...
            if living_nonliving.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                living_nonliving.frameNStart = frameN  # exact frame index
                living_nonliving.tStart = t  # local t and not account for scr refresh
                living_nonliving.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(living_nonliving, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'living_nonliving.started')
                # update status
                living_nonliving.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(living_nonliving.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(living_nonliving.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if living_nonliving is stopping this frame...
            if living_nonliving.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > living_nonliving.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    living_nonliving.tStop = t  # not accounting for scr refresh
                    living_nonliving.tStopRefresh = tThisFlipGlobal  # on global time
                    living_nonliving.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'living_nonliving.stopped')
                    # update status
                    living_nonliving.status = FINISHED
                    living_nonliving.status = FINISHED
            if living_nonliving.status == STARTED and not waitOnFlip:
                theseKeys = living_nonliving.getKeys(keyList=['f','g'], ignoreKeys=["escape"], waitRelease=False)
                _living_nonliving_allKeys.extend(theseKeys)
                if len(_living_nonliving_allKeys):
                    living_nonliving.keys = _living_nonliving_allKeys[-1].name  # just the last key pressed
                    living_nonliving.rt = _living_nonliving_allKeys[-1].rt
                    living_nonliving.duration = _living_nonliving_allKeys[-1].duration
            
            # *end_mapping* updates
            waitOnFlip = False
            
            # if end_mapping is starting this frame...
            if end_mapping.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_mapping.frameNStart = frameN  # exact frame index
                end_mapping.tStart = t  # local t and not account for scr refresh
                end_mapping.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_mapping, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_mapping.started')
                # update status
                end_mapping.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_mapping.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_mapping.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if end_mapping is stopping this frame...
            if end_mapping.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > end_mapping.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    end_mapping.tStop = t  # not accounting for scr refresh
                    end_mapping.tStopRefresh = tThisFlipGlobal  # on global time
                    end_mapping.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'end_mapping.stopped')
                    # update status
                    end_mapping.status = FINISHED
                    end_mapping.status = FINISHED
            if end_mapping.status == STARTED and not waitOnFlip:
                theseKeys = end_mapping.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _end_mapping_allKeys.extend(theseKeys)
                if len(_end_mapping_allKeys):
                    end_mapping.keys = _end_mapping_allKeys[-1].name  # just the last key pressed
                    end_mapping.rt = _end_mapping_allKeys[-1].rt
                    end_mapping.duration = _end_mapping_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *licing_nonliving_text* updates
            
            # if licing_nonliving_text is starting this frame...
            if licing_nonliving_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                licing_nonliving_text.frameNStart = frameN  # exact frame index
                licing_nonliving_text.tStart = t  # local t and not account for scr refresh
                licing_nonliving_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(licing_nonliving_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'licing_nonliving_text.started')
                # update status
                licing_nonliving_text.status = STARTED
                licing_nonliving_text.setAutoDraw(True)
            
            # if licing_nonliving_text is active this frame...
            if licing_nonliving_text.status == STARTED:
                # update params
                pass
            
            # if licing_nonliving_text is stopping this frame...
            if licing_nonliving_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > licing_nonliving_text.tStartRefresh + 5.0-frameTolerance:
                    # keep track of stop time/frame for later
                    licing_nonliving_text.tStop = t  # not accounting for scr refresh
                    licing_nonliving_text.tStopRefresh = tThisFlipGlobal  # on global time
                    licing_nonliving_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'licing_nonliving_text.stopped')
                    # update status
                    licing_nonliving_text.status = FINISHED
                    licing_nonliving_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                item.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in item.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "item" ---
        for thisComponent in item.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for item
        item.tStop = globalClock.getTime(format='float')
        item.tStopRefresh = tThisFlipGlobal
        thisExp.addData('item.stopped', item.tStop)
        # check responses
        if living_nonliving.keys in ['', [], None]:  # No response was made
            living_nonliving.keys = None
        semantic_mapping.addData('living_nonliving.keys',living_nonliving.keys)
        if living_nonliving.keys != None:  # we had a response
            semantic_mapping.addData('living_nonliving.rt', living_nonliving.rt)
            semantic_mapping.addData('living_nonliving.duration', living_nonliving.duration)
        # check responses
        if end_mapping.keys in ['', [], None]:  # No response was made
            end_mapping.keys = None
        semantic_mapping.addData('end_mapping.keys',end_mapping.keys)
        if end_mapping.keys != None:  # we had a response
            semantic_mapping.addData('end_mapping.rt', end_mapping.rt)
            semantic_mapping.addData('end_mapping.duration', end_mapping.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if item.maxDurationReached:
            routineTimer.addTime(-item.maxDuration)
        elif item.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-5.000000)
        
        # --- Prepare to start Routine "mapping_break" ---
        # create an object to store info about Routine mapping_break
        mapping_break = data.Routine(
            name='mapping_break',
            components=[break_text, break_resp],
        )
        mapping_break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for break_resp
        break_resp.keys = []
        break_resp.rt = []
        _break_resp_allKeys = []
        # Run 'Begin Routine' code from reset_counter
        mapping_trial = 0
        # store start times for mapping_break
        mapping_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        mapping_break.tStart = globalClock.getTime(format='float')
        mapping_break.status = STARTED
        thisExp.addData('mapping_break.started', mapping_break.tStart)
        mapping_break.maxDuration = None
        # skip Routine mapping_break if its 'Skip if' condition is True
        mapping_break.skipped = continueRoutine and not (mapping_trial < 50)
        continueRoutine = mapping_break.skipped
        # keep track of which components have finished
        mapping_breakComponents = mapping_break.components
        for thisComponent in mapping_break.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "mapping_break" ---
        # if trial has changed, end Routine now
        if isinstance(semantic_mapping, data.TrialHandler2) and thisSemantic_mapping.thisN != semantic_mapping.thisTrial.thisN:
            continueRoutine = False
        mapping_break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *break_text* updates
            
            # if break_text is starting this frame...
            if break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_text.frameNStart = frameN  # exact frame index
                break_text.tStart = t  # local t and not account for scr refresh
                break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_text.started')
                # update status
                break_text.status = STARTED
                break_text.setAutoDraw(True)
            
            # if break_text is active this frame...
            if break_text.status == STARTED:
                # update params
                pass
            
            # *break_resp* updates
            waitOnFlip = False
            
            # if break_resp is starting this frame...
            if break_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                break_resp.frameNStart = frameN  # exact frame index
                break_resp.tStart = t  # local t and not account for scr refresh
                break_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(break_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'break_resp.started')
                # update status
                break_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(break_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(break_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if break_resp.status == STARTED and not waitOnFlip:
                theseKeys = break_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _break_resp_allKeys.extend(theseKeys)
                if len(_break_resp_allKeys):
                    break_resp.keys = _break_resp_allKeys[-1].name  # just the last key pressed
                    break_resp.rt = _break_resp_allKeys[-1].rt
                    break_resp.duration = _break_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                mapping_break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in mapping_break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "mapping_break" ---
        for thisComponent in mapping_break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for mapping_break
        mapping_break.tStop = globalClock.getTime(format='float')
        mapping_break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('mapping_break.stopped', mapping_break.tStop)
        # check responses
        if break_resp.keys in ['', [], None]:  # No response was made
            break_resp.keys = None
        semantic_mapping.addData('break_resp.keys',break_resp.keys)
        if break_resp.keys != None:  # we had a response
            semantic_mapping.addData('break_resp.rt', break_resp.rt)
            semantic_mapping.addData('break_resp.duration', break_resp.duration)
        # the Routine "mapping_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'semantic_mapping'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "thanks" ---
    # create an object to store info about Routine thanks
    thanks = data.Routine(
        name='thanks',
        components=[thanks_text],
    )
    thanks.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for thanks
    thanks.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    thanks.tStart = globalClock.getTime(format='float')
    thanks.status = STARTED
    thisExp.addData('thanks.started', thanks.tStart)
    thanks.maxDuration = None
    # keep track of which components have finished
    thanksComponents = thanks.components
    for thisComponent in thanks.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "thanks" ---
    thanks.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thanks_text* updates
        
        # if thanks_text is starting this frame...
        if thanks_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thanks_text.frameNStart = frameN  # exact frame index
            thanks_text.tStart = t  # local t and not account for scr refresh
            thanks_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thanks_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thanks_text.started')
            # update status
            thanks_text.status = STARTED
            thanks_text.setAutoDraw(True)
        
        # if thanks_text is active this frame...
        if thanks_text.status == STARTED:
            # update params
            pass
        
        # if thanks_text is stopping this frame...
        if thanks_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > thanks_text.tStartRefresh + 5.0-frameTolerance:
                # keep track of stop time/frame for later
                thanks_text.tStop = t  # not accounting for scr refresh
                thanks_text.tStopRefresh = tThisFlipGlobal  # on global time
                thanks_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'thanks_text.stopped')
                # update status
                thanks_text.status = FINISHED
                thanks_text.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            thanks.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in thanks.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "thanks" ---
    for thisComponent in thanks.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for thanks
    thanks.tStop = globalClock.getTime(format='float')
    thanks.tStopRefresh = tThisFlipGlobal
    thisExp.addData('thanks.stopped', thanks.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if thanks.maxDurationReached:
        routineTimer.addTime(-thanks.maxDuration)
    elif thanks.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
