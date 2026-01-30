import React, { useState, useEffect, useRef, useCallback } from 'react'
import { Mic, MicOff, Play, Square, User, Bot, Volume2, Activity, History, ChevronDown, Settings } from 'lucide-react'

// opus-recorder is a UMD module - need to handle both default and module exports
import RecorderModule from 'opus-recorder'
const Recorder = RecorderModule.default || RecorderModule

// Binary protocol message types
const MSG_HANDSHAKE = 0x00
const MSG_AUDIO = 0x01
const MSG_TEXT = 0x02
const MSG_CONTROL = 0x03
const MSG_METADATA = 0x04
const MSG_ERROR = 0x05

// Voice presets
const VOICE_PRESETS = [
  { id: 'NATF0', name: 'Natural Female 0', type: 'Natural', gender: 'Female' },
  { id: 'NATF1', name: 'Natural Female 1', type: 'Natural', gender: 'Female' },
  { id: 'NATF2', name: 'Natural Female 2', type: 'Natural', gender: 'Female' },
  { id: 'NATF3', name: 'Natural Female 3', type: 'Natural', gender: 'Female' },
  { id: 'NATM0', name: 'Natural Male 0', type: 'Natural', gender: 'Male' },
  { id: 'NATM1', name: 'Natural Male 1', type: 'Natural', gender: 'Male' },
  { id: 'NATM2', name: 'Natural Male 2', type: 'Natural', gender: 'Male' },
  { id: 'NATM3', name: 'Natural Male 3', type: 'Natural', gender: 'Male' },
  { id: 'VARF0', name: 'Variety Female 0', type: 'Variety', gender: 'Female' },
  { id: 'VARF1', name: 'Variety Female 1', type: 'Variety', gender: 'Female' },
  { id: 'VARF2', name: 'Variety Female 2', type: 'Variety', gender: 'Female' },
  { id: 'VARF3', name: 'Variety Female 3', type: 'Variety', gender: 'Female' },
  { id: 'VARF4', name: 'Variety Female 4', type: 'Variety', gender: 'Female' },
  { id: 'VARM0', name: 'Variety Male 0', type: 'Variety', gender: 'Male' },
  { id: 'VARM1', name: 'Variety Male 1', type: 'Variety', gender: 'Male' },
  { id: 'VARM2', name: 'Variety Male 2', type: 'Variety', gender: 'Male' },
  { id: 'VARM3', name: 'Variety Male 3', type: 'Variety', gender: 'Male' },
  { id: 'VARM4', name: 'Variety Male 4', type: 'Variety', gender: 'Male' },
]

// Persona presets
const PERSONA_PRESETS = [
  {
    name: 'Helpful Assistant',
    prompt: 'You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.'
  },
  {
    name: 'Casual Conversation',
    prompt: 'You enjoy having a good conversation.'
  },
  {
    name: 'Customer Service',
    prompt: 'You work for a customer service department. Be polite, helpful, and professional.'
  },
  {
    name: 'Creative Writer',
    prompt: 'You are a creative writer with a vivid imagination. Tell engaging stories and use descriptive language.'
  }
]

// Decode incoming binary message
function decodeMessage(data) {
  const type = data[0]
  const payload = data.slice(1)

  switch (type) {
    case MSG_HANDSHAKE:
      return { type: 'handshake' }
    case MSG_AUDIO:
      return { type: 'audio', data: payload }
    case MSG_TEXT:
      return { type: 'text', data: new TextDecoder().decode(payload) }
    case MSG_CONTROL:
      return { type: 'control', action: payload[0] }
    case MSG_METADATA:
      return { type: 'metadata', data: JSON.parse(new TextDecoder().decode(payload)) }
    case MSG_ERROR:
      return { type: 'error', data: new TextDecoder().decode(payload) }
    default:
      console.warn('Unknown message type:', type)
      return { type: 'unknown', raw: data }
  }
}

// Encode outgoing audio message
function encodeAudioMessage(audioData) {
  const result = new Uint8Array(1 + audioData.length)
  result[0] = MSG_AUDIO
  result.set(audioData, 1)
  return result
}

// Compact Voice Dropdown Component
function VoiceDropdown({ selectedVoice, setSelectedVoice, compact = false, disabled = false }) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef(null)

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const selectedVoiceData = VOICE_PRESETS.find(v => v.id === selectedVoice)

  return (
    <div ref={dropdownRef} style={{ position: 'relative', marginBottom: compact ? '0.5rem' : '1rem' }}>
      <label style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block', marginBottom: '0.25rem' }}>
        Voice
      </label>
      <button
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        style={{
          width: '100%',
          padding: '0.5rem 0.75rem',
          background: disabled ? '#1e293b' : '#0f172a',
          color: disabled ? '#64748b' : '#e2e8f0',
          border: '1px solid #334155',
          borderRadius: '6px',
          cursor: disabled ? 'not-allowed' : 'pointer',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: '0.875rem'
        }}
      >
        <span>
          {selectedVoiceData ? (
            <>
              {selectedVoiceData.name}
              <span style={{ color: '#64748b', marginLeft: '0.5rem', fontSize: '0.75rem' }}>
                ({selectedVoiceData.type} - {selectedVoiceData.gender})
              </span>
            </>
          ) : 'Select a voice...'}
        </span>
        <ChevronDown size={16} style={{ transform: isOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
      </button>

      {isOpen && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          right: 0,
          marginTop: '0.25rem',
          background: '#1e293b',
          border: '1px solid #334155',
          borderRadius: '6px',
          maxHeight: '200px',
          overflowY: 'auto',
          zIndex: 100,
          boxShadow: '0 4px 12px rgba(0,0,0,0.3)'
        }}>
          {VOICE_PRESETS.map(voice => (
            <div
              key={voice.id}
              onClick={() => {
                setSelectedVoice(voice.id)
                setIsOpen(false)
              }}
              style={{
                padding: '0.5rem 0.75rem',
                cursor: 'pointer',
                background: selectedVoice === voice.id ? '#1e40af' : 'transparent',
                borderBottom: '1px solid #334155'
              }}
              onMouseEnter={(e) => e.target.style.background = selectedVoice === voice.id ? '#1e40af' : '#334155'}
              onMouseLeave={(e) => e.target.style.background = selectedVoice === voice.id ? '#1e40af' : 'transparent'}
            >
              <div style={{ fontSize: '0.875rem' }}>{voice.name}</div>
              <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>{voice.type} - {voice.gender}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Compact Persona Input Component
function PersonaInput({ persona, setPersona, compact = false, disabled = false }) {
  const [expanded, setExpanded] = useState(false)
  const [selectedPreset, setSelectedPreset] = useState(null)

  const applyPreset = (preset) => {
    setSelectedPreset(preset.name)
    setPersona(preset.prompt)
  }

  return (
    <div style={{ marginBottom: compact ? '0.5rem' : '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
        <label style={{ fontSize: '0.75rem', color: '#94a3b8' }}>Persona</label>
        {!expanded && persona && (
          <button
            onClick={() => setExpanded(true)}
            disabled={disabled}
            style={{
              fontSize: '0.7rem',
              color: '#60a5fa',
              background: 'none',
              border: 'none',
              cursor: disabled ? 'not-allowed' : 'pointer',
              padding: 0
            }}
          >
            Expand
          </button>
        )}
      </div>

      {/* Preset chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem', marginBottom: '0.5rem' }}>
        {PERSONA_PRESETS.map(preset => (
          <button
            key={preset.name}
            onClick={() => !disabled && applyPreset(preset)}
            disabled={disabled}
            style={{
              padding: '0.2rem 0.5rem',
              background: selectedPreset === preset.name ? '#1e40af' : '#334155',
              color: disabled ? '#64748b' : 'white',
              border: 'none',
              borderRadius: '3px',
              cursor: disabled ? 'not-allowed' : 'pointer',
              fontSize: '0.65rem'
            }}
          >
            {preset.name}
          </button>
        ))}
      </div>

      <textarea
        value={persona}
        onChange={(e) => {
          setPersona(e.target.value)
          setSelectedPreset(null)
        }}
        disabled={disabled}
        placeholder="Describe persona..."
        style={{
          width: '100%',
          minHeight: expanded ? '100px' : '40px',
          padding: '0.5rem',
          background: disabled ? '#1e293b' : '#0f172a',
          color: disabled ? '#64748b' : '#e2e8f0',
          border: '1px solid #334155',
          borderRadius: '6px',
          fontSize: '0.75rem',
          resize: 'vertical',
          fontFamily: 'inherit'
        }}
      />
      {expanded && (
        <button
          onClick={() => setExpanded(false)}
          style={{
            fontSize: '0.7rem',
            color: '#60a5fa',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            padding: 0,
            marginTop: '0.25rem'
          }}
        >
          Collapse
        </button>
      )}
    </div>
  )
}

// Compact Mic Settings Component
function MicSettings({ settings, setSettings, compact = false, disabled = false }) {
  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '0.75rem',
      padding: '0.5rem 0',
      marginBottom: compact ? '0.5rem' : '1rem',
      flexWrap: 'wrap'
    }}>
      {/* Sensitivity slider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
        <label style={{ fontSize: '0.65rem', color: '#94a3b8' }}>Sens</label>
        <input
          type="range"
          min="0.5"
          max="2"
          step="0.1"
          value={settings.sensitivity}
          onChange={(e) => setSettings({ ...settings, sensitivity: parseFloat(e.target.value) })}
          disabled={disabled}
          style={{ width: '50px', height: '4px' }}
        />
        <span style={{ fontSize: '0.6rem', color: '#64748b', width: '24px' }}>{settings.sensitivity.toFixed(1)}</span>
      </div>

      {/* Noise Suppression toggle */}
      <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', cursor: disabled ? 'not-allowed' : 'pointer' }}>
        <input
          type="checkbox"
          checked={settings.noiseSuppression}
          onChange={(e) => setSettings({ ...settings, noiseSuppression: e.target.checked })}
          disabled={disabled}
          style={{ width: '12px', height: '12px' }}
        />
        <span style={{ fontSize: '0.65rem', color: '#94a3b8' }} title="Noise Suppression">NS</span>
      </label>

      {/* Echo Cancellation toggle */}
      <label style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', cursor: disabled ? 'not-allowed' : 'pointer' }}>
        <input
          type="checkbox"
          checked={settings.echoCancellation}
          onChange={(e) => setSettings({ ...settings, echoCancellation: e.target.checked })}
          disabled={disabled}
          style={{ width: '12px', height: '12px' }}
        />
        <span style={{ fontSize: '0.65rem', color: '#94a3b8' }} title="Echo Cancellation">EC</span>
      </label>
    </div>
  )
}

// Audio Level Meter Component
function AudioLevelMeter({ level, compact = false }) {
  return (
    <div style={{
      padding: '0.5rem',
      background: '#0f172a',
      borderRadius: '4px',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem'
    }}>
      <Mic size={14} color="#94a3b8" />
      <div style={{
        flex: 1,
        height: '6px',
        background: '#1e293b',
        borderRadius: '3px',
        overflow: 'hidden'
      }}>
        <div style={{
          height: '100%',
          width: `${(level / 255) * 100}%`,
          background: level > 200 ? '#ef4444' : level > 100 ? '#eab308' : '#22c55e',
          transition: 'width 0.05s ease-out'
        }} />
      </div>
    </div>
  )
}

// Voice Selection Screen (legacy - keeping for non-compact mode)
function VoiceSelection({ selectedVoice, setSelectedVoice, onNext }) {
  return (
    <div style={{ padding: '2rem' }}>
      <h2 style={{ fontSize: '1.5rem', marginBottom: '1.5rem', color: '#60a5fa' }}>
        Select a Voice
      </h2>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '1rem' }}>
        {VOICE_PRESETS.map(voice => (
          <div
            key={voice.id}
            onClick={() => setSelectedVoice(voice.id)}
            style={{
              padding: '1rem',
              borderRadius: '8px',
              cursor: 'pointer',
              background: selectedVoice === voice.id ? '#1e40af' : '#1e293b',
              border: selectedVoice === voice.id ? '2px solid #60a5fa' : '2px solid transparent',
              transition: 'all 0.2s'
            }}
          >
            <div style={{ fontWeight: 'bold', marginBottom: '0.25rem' }}>
              {voice.name}
            </div>
            <div style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
              {voice.type} - {voice.gender}
            </div>
          </div>
        ))}
      </div>

      <button
        onClick={onNext}
        disabled={!selectedVoice}
        style={{
          marginTop: '2rem',
          padding: '0.75rem 2rem',
          background: selectedVoice ? '#2563eb' : '#475569',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          cursor: selectedVoice ? 'pointer' : 'not-allowed',
          fontSize: '1rem'
        }}
      >
        Continue
      </button>
    </div>
  )
}

// Persona Editor Screen
function PersonaEditor({ persona, setPersona, onBack, onStart }) {
  const [selectedPreset, setSelectedPreset] = useState(null)
  const [customPrompt, setCustomPrompt] = useState(persona)

  const applyPreset = (preset) => {
    setSelectedPreset(preset.name)
    setCustomPrompt(preset.prompt)
    setPersona(preset.prompt)
  }

  const handleCustomChange = (e) => {
    setCustomPrompt(e.target.value)
    setPersona(e.target.value)
    setSelectedPreset(null)
  }

  return (
    <div style={{ padding: '2rem' }}>
      <h2 style={{ fontSize: '1.5rem', marginBottom: '1.5rem', color: '#60a5fa' }}>
        Define the Persona
      </h2>

      <div style={{ marginBottom: '1.5rem' }}>
        <h3 style={{ fontSize: '1rem', marginBottom: '0.75rem', color: '#94a3b8' }}>
          Quick Presets:
        </h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {PERSONA_PRESETS.map(preset => (
            <button
              key={preset.name}
              onClick={() => applyPreset(preset)}
              style={{
                padding: '0.5rem 1rem',
                background: selectedPreset === preset.name ? '#1e40af' : '#334155',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '0.875rem'
              }}
            >
              {preset.name}
            </button>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: '1.5rem' }}>
        <h3 style={{ fontSize: '1rem', marginBottom: '0.5rem', color: '#94a3b8' }}>
          Custom Persona Prompt:
        </h3>
        <textarea
          value={customPrompt}
          onChange={handleCustomChange}
          placeholder="Describe the persona's personality, background, and behavior..."
          style={{
            width: '100%',
            minHeight: '150px',
            padding: '0.75rem',
            background: '#1e293b',
            color: '#e2e8f0',
            border: '1px solid #475569',
            borderRadius: '6px',
            fontSize: '0.875rem',
            resize: 'vertical'
          }}
        />
      </div>

      <div style={{ display: 'flex', gap: '1rem' }}>
        <button
          onClick={onBack}
          style={{
            padding: '0.75rem 2rem',
            background: '#475569',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '1rem'
          }}
        >
          Back
        </button>
        <button
          onClick={onStart}
          disabled={!persona.trim()}
          style={{
            padding: '0.75rem 2rem',
            background: persona.trim() ? '#2563eb' : '#475569',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: persona.trim() ? 'pointer' : 'not-allowed',
            fontSize: '1rem'
          }}
        >
          Start Conversation
        </button>
      </div>
    </div>
  )
}

// Active Conversation Screen
function ActiveConversation({ voice, persona, onEnd, serverUrl, micSettings }) {
  const [status, setStatus] = useState('connecting')  // connecting, priming, ready, recording, error
  const [transcript, setTranscript] = useState([])
  const [currentText, setCurrentText] = useState('')
  const [audioLevel, setAudioLevel] = useState(0)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [errorMessage, setErrorMessage] = useState(null)

  // Ref for synchronous text accumulation (fixes disappearing text bug)
  const currentTextRef = useRef('')

  const socketRef = useRef(null)
  const recorderRef = useRef(null)
  const audioContextRef = useRef(null)
  const workletRef = useRef(null)
  const decoderWorkerRef = useRef(null)
  const analyserRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const gainNodeRef = useRef(null)
  const micDurationRef = useRef(0)
  const animationFrameRef = useRef(null)
  const transcriptEndRef = useRef(null)

  // Auto-scroll transcript
  useEffect(() => {
    if (transcriptEndRef.current) {
      transcriptEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [transcript, currentText])

  // Initialize decoder worker
  const initDecoderWorker = useCallback(() => {
    return new Promise((resolve) => {
      const worker = new Worker(new URL('/assets/decoderWorker.min.js', import.meta.url))

      worker.onmessage = (e) => {
        if (!e.data) return

        // Decoded audio data
        const audioData = e.data[0]
        if (audioData && workletRef.current) {
          workletRef.current.port.postMessage({
            frame: audioData,
            type: 'audio',
            micDuration: micDurationRef.current
          })
          setIsSpeaking(true)
        }
      }

      worker.onerror = (e) => {
        console.error('Decoder worker error:', e)
      }

      decoderWorkerRef.current = worker

      // Initialize the decoder
      const sampleRate = audioContextRef.current?.sampleRate || 48000
      worker.postMessage({
        command: 'init',
        bufferLength: Math.round(960 * sampleRate / 24000),
        decoderSampleRate: 24000,
        outputBufferSampleRate: sampleRate,
        resampleQuality: 0
      })

      // Send warmup BOS page after a delay
      setTimeout(() => {
        const bosPage = createWarmupBosPage()
        worker.postMessage({
          command: 'decode',
          pages: bosPage
        })
      }, 100)

      // Give decoder time to initialize
      setTimeout(resolve, 500)
    })
  }, [])

  // Create warmup BOS page for decoder
  function createWarmupBosPage() {
    const opusHead = new Uint8Array([
      0x4F, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64, // "OpusHead"
      0x01,       // Version 1
      0x01,       // 1 channel (mono)
      0x38, 0x01, // Pre-skip: 312 samples
      0x80, 0xBB, 0x00, 0x00, // Sample rate: 48000 Hz
      0x00, 0x00, // Output gain: 0
      0x00,       // Channel mapping: 0
    ])

    const pageHeader = new Uint8Array([
      0x4F, 0x67, 0x67, 0x53, // "OggS"
      0x00,       // Version 0
      0x02,       // BOS flag
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Granule position
      0x01, 0x00, 0x00, 0x00, // Stream serial
      0x00, 0x00, 0x00, 0x00, // Page sequence
      0x00, 0x00, 0x00, 0x00, // CRC
      0x01,       // 1 segment
      0x13,       // Segment size: 19 bytes
    ])

    const bosPage = new Uint8Array(pageHeader.length + opusHead.length)
    bosPage.set(pageHeader, 0)
    bosPage.set(opusHead, pageHeader.length)
    return bosPage
  }

  // Initialize audio context and worklet
  const initAudioContext = useCallback(async () => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 })
    audioContextRef.current = audioContext

    // Load and register the audio worklet
    const processorCode = `
      class MoshiProcessor extends AudioWorkletProcessor {
        constructor() {
          super()
          this.frames = []
          this.offsetInFirstBuffer = 0
          this.started = false
          this.remainingPartialBufferSamples = 0
          this.initialBufferSamples = Math.round(80 * sampleRate / 1000)
          this.partialBufferSamples = Math.round(10 * sampleRate / 1000)

          this.port.onmessage = (event) => {
            if (event.data.type === 'reset') {
              this.frames = []
              this.offsetInFirstBuffer = 0
              this.started = false
              return
            }
            this.frames.push(event.data.frame)
            if (this.currentSamples() >= this.initialBufferSamples && !this.started) {
              this.started = true
              this.remainingPartialBufferSamples = this.partialBufferSamples
            }
          }
        }

        currentSamples() {
          let samples = 0
          for (let i = 0; i < this.frames.length; i++) {
            samples += this.frames[i].length
          }
          return samples - this.offsetInFirstBuffer
        }

        canPlay() {
          return this.started && this.frames.length > 0 && this.remainingPartialBufferSamples <= 0
        }

        process(inputs, outputs) {
          const output = outputs[0][0]
          this.remainingPartialBufferSamples -= output.length

          if (!this.canPlay()) {
            return true
          }

          let outIdx = 0
          while (outIdx < output.length && this.frames.length) {
            const first = this.frames[0]
            const toCopy = Math.min(first.length - this.offsetInFirstBuffer, output.length - outIdx)
            output.set(first.subarray(this.offsetInFirstBuffer, this.offsetInFirstBuffer + toCopy), outIdx)
            this.offsetInFirstBuffer += toCopy
            outIdx += toCopy
            if (this.offsetInFirstBuffer === first.length) {
              this.offsetInFirstBuffer = 0
              this.frames.shift()
            }
          }

          if (outIdx < output.length) {
            this.started = false
          }

          return true
        }
      }
      registerProcessor('moshi-processor', MoshiProcessor)
    `

    const blob = new Blob([processorCode], { type: 'application/javascript' })
    const url = URL.createObjectURL(blob)

    await audioContext.audioWorklet.addModule(url)
    URL.revokeObjectURL(url)

    const workletNode = new AudioWorkletNode(audioContext, 'moshi-processor')
    workletNode.connect(audioContext.destination)
    workletRef.current = workletNode

    return audioContext
  }, [])

  // Start recording with opus-recorder
  const startRecording = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 24000,
        channelCount: 1,
        echoCancellation: micSettings?.echoCancellation ?? true,
        noiseSuppression: micSettings?.noiseSuppression ?? true,
        autoGainControl: true
      }
    })
    mediaStreamRef.current = stream

    // Setup analyzer for audio level meter with GainNode for sensitivity
    const source = audioContextRef.current.createMediaStreamSource(stream)

    // Create gain node for sensitivity control
    const gainNode = audioContextRef.current.createGain()
    gainNode.gain.value = micSettings?.sensitivity ?? 1.0
    gainNodeRef.current = gainNode

    const analyser = audioContextRef.current.createAnalyser()
    analyser.fftSize = 256

    // Connect: source -> gain -> analyser
    source.connect(gainNode)
    gainNode.connect(analyser)
    analyserRef.current = analyser

    // Audio level animation
    const updateLevel = () => {
      if (!analyserRef.current) return

      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
      analyserRef.current.getByteFrequencyData(dataArray)
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length
      setAudioLevel(average)

      animationFrameRef.current = requestAnimationFrame(updateLevel)
    }
    updateLevel()

    // Setup opus-recorder with streaming
    const audioContextSampleRate = audioContextRef.current.sampleRate
    const recorder = new Recorder({
      encoderPath: new URL('/assets/encoderWorker.min.js', import.meta.url).href,
      bufferLength: Math.round(960 * audioContextSampleRate / 24000),
      encoderFrameSize: 20,
      encoderSampleRate: 24000,
      maxFramesPerPage: 2,
      numberOfChannels: 1,
      recordingGain: micSettings?.sensitivity ?? 1,
      resampleQuality: 3,
      encoderComplexity: 0,
      encoderApplication: 2049,
      streamPages: true,
      mediaTrackConstraints: {
        echoCancellation: micSettings?.echoCancellation ?? true,
        noiseSuppression: micSettings?.noiseSuppression ?? true,
        autoGainControl: true
      }
    })

    recorder.ondataavailable = (data) => {
      micDurationRef.current = recorder.encodedSamplePosition / 48000

      // Send audio with protocol prefix
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        const message = encodeAudioMessage(data)
        socketRef.current.send(message)
      }
    }

    recorder.onstart = () => {
      setStatus('recording')
    }

    recorder.onstop = () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
      }
    }

    recorderRef.current = recorder
    recorder.start(stream)
  }, [])

  // Stop recording
  const stopRecording = useCallback(() => {
    if (recorderRef.current) {
      recorderRef.current.stop()
      recorderRef.current = null
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current)
    }

    setAudioLevel(0)
  }, [])

  // Connect and setup WebSocket
  useEffect(() => {
    let mounted = true

    async function setup() {
      // Initialize audio context first
      await initAudioContext()

      // Initialize decoder
      await initDecoderWorker()

      if (!mounted) return

      // Build WebSocket URL with query parameters
      const params = new URLSearchParams({
        voice_prompt: `${voice}.pt`,
        text_prompt: persona
      })
      const wsUrl = `ws://${serverUrl}/api/chat?${params.toString()}`

      const socket = new WebSocket(wsUrl)
      socket.binaryType = 'arraybuffer'
      socketRef.current = socket

      socket.onopen = () => {
        if (!mounted) return
        setStatus('priming')
      }

      socket.onmessage = (event) => {
        if (!mounted) return

        const data = new Uint8Array(event.data)
        const message = decodeMessage(data)

        switch (message.type) {
          case 'handshake':
            // Server is ready, start recording
            setStatus('ready')
            startRecording()
            break

          case 'audio':
            // Decode and play audio
            if (decoderWorkerRef.current) {
              decoderWorkerRef.current.postMessage({
                command: 'decode',
                pages: message.data
              }, [message.data.buffer])
            }
            break

          case 'text':
            // Accumulate text using ref for synchronous access
            currentTextRef.current += message.data
            setCurrentText(currentTextRef.current)

            // Check for sentence end using ref (not stale state)
            if (currentTextRef.current.match(/[.!?]\s*$/)) {
              setTranscript(prev => [...prev, {
                speaker: 'ai',
                text: currentTextRef.current.trim(),
                timestamp: new Date()
              }])
              currentTextRef.current = ''
              setCurrentText('')
            }
            setIsSpeaking(true)
            break

          case 'error':
            setErrorMessage(message.data)
            setStatus('error')
            break

          default:
            break
        }
      }

      socket.onclose = () => {
        if (!mounted) return
        setStatus('connecting')
        stopRecording()
      }

      socket.onerror = (err) => {
        console.error('WebSocket error:', err)
        setErrorMessage('Connection error')
        setStatus('error')
      }
    }

    setup()

    return () => {
      mounted = false

      stopRecording()

      if (socketRef.current) {
        socketRef.current.close()
        socketRef.current = null
      }

      if (decoderWorkerRef.current) {
        decoderWorkerRef.current.terminate()
        decoderWorkerRef.current = null
      }

      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close()
      }
    }
  }, [voice, persona, serverUrl, initAudioContext, initDecoderWorker, startRecording, stopRecording])

  // Handle end conversation
  const handleEnd = useCallback(async () => {
    stopRecording()
    if (socketRef.current) {
      socketRef.current.close()
    }

    // Offer to save conversation if there's content
    if (transcript.length > 0) {
      const shouldSave = window.confirm('Save this conversation to history?')
      if (shouldSave) {
        try {
          await fetch(`http://${serverUrl}/api/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              voice_preset: voice,
              persona_prompt: persona,
              transcript: transcript
            })
          })
        } catch (err) {
          console.error('Failed to save conversation:', err)
        }
      }
    }

    onEnd()
  }, [stopRecording, onEnd, transcript, serverUrl, voice, persona])

  // Get status display text
  const getStatusText = () => {
    switch (status) {
      case 'connecting': return 'Connecting to server...'
      case 'priming': return 'Preparing voice model...'
      case 'ready': return 'Starting microphone...'
      case 'recording': return 'Listening'
      case 'error': return `Error: ${errorMessage}`
      default: return status
    }
  }

  return (
    <div style={{ padding: '2rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header Info */}
      <div style={{
        marginBottom: '1rem',
        padding: '1rem',
        background: '#1e293b',
        borderRadius: '8px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <div>
          <div style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
            Voice: {voice}
          </div>
          <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
            Persona: {persona.substring(0, 100)}...
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {status === 'recording' && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.25rem 0.75rem',
              background: '#0f172a',
              borderRadius: '4px'
            }}>
              <Activity size={16} color="#22c55e" />
              <span style={{ fontSize: '0.75rem', color: '#22c55e' }}>Listening</span>
            </div>
          )}
          {isSpeaking && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.25rem 0.75rem',
              background: '#0f172a',
              borderRadius: '4px'
            }}>
              <Volume2 size={16} color="#60a5fa" />
              <span style={{ fontSize: '0.75rem', color: '#60a5fa' }}>Speaking</span>
            </div>
          )}
        </div>
      </div>

      {/* Status Message */}
      {status !== 'recording' && (
        <div style={{
          marginBottom: '1rem',
          padding: '1rem',
          background: status === 'error' ? '#450a0a' : '#0f172a',
          borderRadius: '8px',
          textAlign: 'center',
          color: status === 'error' ? '#fca5a5' : '#94a3b8'
        }}>
          {getStatusText()}
        </div>
      )}

      {/* Audio Level Meter */}
      {status === 'recording' && (
        <div style={{
          marginBottom: '1rem',
          padding: '0.75rem 1rem',
          background: '#0f172a',
          borderRadius: '8px',
          display: 'flex',
          alignItems: 'center',
          gap: '1rem'
        }}>
          <span style={{ fontSize: '0.75rem', color: '#94a3b8', minWidth: '50px' }}>
            Input
          </span>
          <div style={{
            flex: 1,
            height: '8px',
            background: '#1e293b',
            borderRadius: '4px',
            overflow: 'hidden'
          }}>
            <div style={{
              height: '100%',
              width: `${(audioLevel / 255) * 100}%`,
              background: audioLevel > 200 ? '#ef4444' : audioLevel > 100 ? '#eab308' : '#22c55e',
              transition: 'width 0.05s ease-out'
            }} />
          </div>
        </div>
      )}

      {/* Transcript Area */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        marginBottom: '1.5rem',
        padding: '1rem',
        background: '#0f172a',
        borderRadius: '8px'
      }}>
        {transcript.length === 0 && !currentText ? (
          <div style={{ textAlign: 'center', color: '#64748b', padding: '2rem' }}>
            {status === 'recording'
              ? 'Listening... Speak naturally. The AI will respond in real-time.'
              : 'Waiting for connection...'
            }
          </div>
        ) : (
          <>
            {transcript.map((msg, idx) => (
              <div
                key={idx}
                style={{
                  marginBottom: '1rem',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.75rem'
                }}
              >
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: msg.speaker === 'user' ? '#2563eb' : '#059669',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0
                }}>
                  {msg.speaker === 'user' ? <User size={16} /> : <Bot size={16} />}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{
                    fontSize: '0.75rem',
                    color: '#94a3b8',
                    marginBottom: '0.25rem',
                    display: 'flex',
                    justifyContent: 'space-between'
                  }}>
                    <span>{msg.speaker === 'user' ? 'You' : 'PersonaPlex'}</span>
                    <span>{msg.timestamp?.toLocaleTimeString?.() || ''}</span>
                  </div>
                  <div style={{ lineHeight: '1.5' }}>{msg.text}</div>
                </div>
              </div>
            ))}

            {/* Current AI text (streaming) */}
            {currentText && (
              <div style={{
                marginBottom: '1rem',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.75rem'
              }}>
                <div style={{
                  width: '32px',
                  height: '32px',
                  borderRadius: '50%',
                  background: '#059669',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  flexShrink: 0
                }}>
                  <Bot size={16} />
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
                    PersonaPlex
                  </div>
                  <div style={{ lineHeight: '1.5' }}>
                    {currentText}
                    <span style={{
                      display: 'inline-block',
                      width: '2px',
                      height: '1em',
                      background: '#60a5fa',
                      marginLeft: '2px',
                      animation: 'blink 1s infinite'
                    }} />
                  </div>
                </div>
              </div>
            )}

            <div ref={transcriptEndRef} />
          </>
        )}
      </div>

      {/* Control Buttons */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem' }}>
        <button
          onClick={handleEnd}
          style={{
            padding: '0.75rem 1.5rem',
            background: '#475569',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontSize: '0.875rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}
        >
          <Square size={16} />
          End Conversation
        </button>
      </div>

      <style>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
      `}</style>
    </div>
  )
}

// History Screen
function HistoryScreen({ onBack, serverUrl }) {
  const [conversations, setConversations] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`http://${serverUrl}/api/history`)
      .then(res => res.json())
      .then(data => {
        setConversations(data.conversations || [])
        setLoading(false)
      })
      .catch(err => {
        console.error('Failed to fetch history:', err)
        setLoading(false)
      })
  }, [serverUrl])

  return (
    <div style={{ padding: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.5rem' }}>
        <button
          onClick={onBack}
          style={{
            padding: '0.5rem 1rem',
            background: '#475569',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Back
        </button>
        <h2 style={{ fontSize: '1.5rem', color: '#60a5fa' }}>
          Conversation History
        </h2>
      </div>

      {loading ? (
        <div style={{ textAlign: 'center', color: '#64748b', padding: '3rem' }}>
          Loading...
        </div>
      ) : conversations.length === 0 ? (
        <div style={{ textAlign: 'center', color: '#64748b', padding: '3rem' }}>
          <History size={48} style={{ marginBottom: '1rem', opacity: 0.5 }} />
          <p>No conversations yet. Start one to see it here!</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {conversations.map(conv => (
            <div
              key={conv.id}
              style={{
                padding: '1rem',
                background: '#1e293b',
                borderRadius: '8px'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                <span style={{ fontWeight: 'bold' }}>{conv.voice_preset}</span>
                <span style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
                  {new Date(conv.created_at).toLocaleDateString()}
                </span>
              </div>
              <div style={{ color: '#94a3b8', fontSize: '0.875rem' }}>
                {conv.persona_preview}...
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// Compact Transcript List Component
function TranscriptList({ transcript, currentText, status }) {
  const transcriptEndRef = useRef(null)

  useEffect(() => {
    if (transcriptEndRef.current) {
      transcriptEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [transcript, currentText])

  return (
    <div style={{
      flex: 1,
      overflowY: 'auto',
      padding: '0.5rem',
      background: '#0f172a',
      borderRadius: '6px'
    }}>
      {transcript.length === 0 && !currentText ? (
        <div style={{ textAlign: 'center', color: '#64748b', padding: '1rem', fontSize: '0.75rem' }}>
          {status === 'recording'
            ? 'Listening... Speak naturally.'
            : status === 'connecting'
            ? 'Connecting...'
            : 'Ready to start'
          }
        </div>
      ) : (
        <>
          {transcript.map((msg, idx) => (
            <div
              key={idx}
              style={{
                marginBottom: '0.75rem',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.5rem'
              }}
            >
              <div style={{
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                background: msg.speaker === 'user' ? '#2563eb' : '#059669',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0
              }}>
                {msg.speaker === 'user' ? <User size={12} /> : <Bot size={12} />}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: '0.65rem',
                  color: '#94a3b8',
                  marginBottom: '0.15rem'
                }}>
                  {msg.speaker === 'user' ? 'You' : 'AI'}
                </div>
                <div style={{ fontSize: '0.8rem', lineHeight: '1.4', wordBreak: 'break-word' }}>{msg.text}</div>
              </div>
            </div>
          ))}

          {currentText && (
            <div style={{
              marginBottom: '0.75rem',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '0.5rem'
            }}>
              <div style={{
                width: '24px',
                height: '24px',
                borderRadius: '50%',
                background: '#059669',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0
              }}>
                <Bot size={12} />
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: '0.65rem', color: '#94a3b8', marginBottom: '0.15rem' }}>AI</div>
                <div style={{ fontSize: '0.8rem', lineHeight: '1.4', wordBreak: 'break-word' }}>
                  {currentText}
                  <span style={{
                    display: 'inline-block',
                    width: '2px',
                    height: '1em',
                    background: '#60a5fa',
                    marginLeft: '2px',
                    animation: 'blink 1s infinite'
                  }} />
                </div>
              </div>
            </div>
          )}

          <div ref={transcriptEndRef} />
        </>
      )}

      <style>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
      `}</style>
    </div>
  )
}

// Main App Component with Compact Left Panel Layout
function App() {
  const [isActive, setIsActive] = useState(false)
  const [selectedVoice, setSelectedVoice] = useState('NATF0')
  const [persona, setPersona] = useState('You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.')
  const [serverStatus, setServerStatus] = useState('checking')
  const [micSettings, setMicSettings] = useState({
    sensitivity: 1.0,
    noiseSuppression: true,
    echoCancellation: true
  })
  const [showHistory, setShowHistory] = useState(false)

  // Server URL - Lambda Labs cloud GPU
  const serverUrl = '150.136.94.234:8080'

  // Check server connection
  useEffect(() => {
    const checkServer = async () => {
      // For cloud servers, skip HTTP check (CORS blocks it) - assume connected
      if (serverUrl.includes('150.136.94.234')) {
        setServerStatus('connected')
        return
      }

      try {
        const response = await fetch(`http://${serverUrl}/`, { method: 'HEAD' })
        if (response.ok) {
          setServerStatus('connected')
          return
        }
      } catch (err) {
        console.log('Server check failed:', err.message)
        setServerStatus('disconnected')
      }
    }

    checkServer()
    const interval = setInterval(checkServer, 10000)
    return () => clearInterval(interval)
  }, [serverUrl])

  // Show history screen
  if (showHistory) {
    return (
      <div style={{ minHeight: '100vh', background: '#0a0a0a' }}>
        <HistoryScreen onBack={() => setShowHistory(false)} serverUrl={serverUrl} />
      </div>
    )
  }

  return (
    <div style={{ height: '100vh', display: 'flex', background: '#0a0a0a' }}>
      {/* LEFT PANEL - 1/3 width */}
      <div style={{
        width: '33.33%',
        minWidth: '320px',
        maxWidth: '400px',
        height: '100vh',
        display: 'flex',
        flexDirection: 'column',
        borderRight: '1px solid #334155',
        background: '#0f172a'
      }}>
        {/* Header */}
        <div style={{
          padding: '0.75rem 1rem',
          borderBottom: '1px solid #334155',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <h1 style={{ fontSize: '1rem', color: '#60a5fa', margin: 0 }}>
            PersonaPlex
          </h1>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              onClick={() => setShowHistory(true)}
              style={{
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: '0.25rem',
                color: '#94a3b8'
              }}
              title="View History"
            >
              <History size={16} />
            </button>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: serverStatus === 'connected' ? '#22c55e' : serverStatus === 'checking' ? '#eab308' : '#ef4444'
            }} />
            <span style={{ fontSize: '0.7rem', color: '#94a3b8' }}>
              {serverStatus === 'connected' ? 'Connected' : serverStatus === 'checking' ? 'Checking...' : 'Offline'}
            </span>
          </div>
        </div>

        {/* Controls Section */}
        <div style={{ padding: '1rem', borderBottom: '1px solid #334155' }}>
          <VoiceDropdown
            selectedVoice={selectedVoice}
            setSelectedVoice={setSelectedVoice}
            compact
            disabled={isActive}
          />

          <PersonaInput
            persona={persona}
            setPersona={setPersona}
            compact
            disabled={isActive}
          />

          <MicSettings
            settings={micSettings}
            setSettings={setMicSettings}
            compact
            disabled={isActive}
          />

          {/* Start/End Button */}
          <button
            onClick={() => setIsActive(!isActive)}
            disabled={serverStatus !== 'connected' || !selectedVoice || !persona.trim()}
            style={{
              width: '100%',
              padding: '0.6rem 1rem',
              background: isActive ? '#dc2626' : (serverStatus === 'connected' && selectedVoice && persona.trim()) ? '#2563eb' : '#475569',
              color: 'white',
              border: 'none',
              borderRadius: '6px',
              cursor: (serverStatus === 'connected' && selectedVoice && persona.trim()) ? 'pointer' : 'not-allowed',
              fontSize: '0.875rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '0.5rem'
            }}
          >
            {isActive ? (
              <>
                <Square size={16} />
                End Conversation
              </>
            ) : (
              <>
                <Play size={16} />
                Start Conversation
              </>
            )}
          </button>
        </div>

        {/* Transcript Section - Fills remaining space */}
        <div style={{ flex: 1, padding: '0.75rem', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          {isActive ? (
            <ActiveConversationInline
              voice={selectedVoice}
              persona={persona}
              serverUrl={serverUrl}
              micSettings={micSettings}
              onEnd={() => setIsActive(false)}
            />
          ) : (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#64748b',
              textAlign: 'center',
              padding: '2rem'
            }}>
              <Bot size={32} style={{ marginBottom: '0.75rem', opacity: 0.5 }} />
              <p style={{ fontSize: '0.8rem', margin: 0 }}>
                Configure voice and persona above, then click Start to begin.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* RIGHT PANEL - 2/3 width (Reserved for future features) */}
      <div style={{
        flex: 1,
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#0a0a0a'
      }}>
        <div style={{ color: '#334155', textAlign: 'center' }}>
          <h2 style={{ fontSize: '1.25rem', marginBottom: '1rem', color: '#475569' }}>Future Features</h2>
          <div style={{ fontSize: '0.875rem', lineHeight: '2' }}>
            <p>Virtual Audio Routing</p>
            <p>Call Analytics</p>
            <p>B2B Integration</p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Inline Active Conversation (for left panel)
function ActiveConversationInline({ voice, persona, serverUrl, micSettings, onEnd }) {
  const [status, setStatus] = useState('connecting')
  const [transcript, setTranscript] = useState([])
  const [currentText, setCurrentText] = useState('')
  const [audioLevel, setAudioLevel] = useState(0)
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [errorMessage, setErrorMessage] = useState(null)

  const currentTextRef = useRef('')
  const socketRef = useRef(null)
  const recorderRef = useRef(null)
  const audioContextRef = useRef(null)
  const workletRef = useRef(null)
  const decoderWorkerRef = useRef(null)
  const analyserRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const gainNodeRef = useRef(null)
  const micDurationRef = useRef(0)
  const animationFrameRef = useRef(null)

  const initDecoderWorker = useCallback(() => {
    return new Promise((resolve) => {
      const worker = new Worker(new URL('/assets/decoderWorker.min.js', import.meta.url))

      worker.onmessage = (e) => {
        if (!e.data) return
        const audioData = e.data[0]
        if (audioData && workletRef.current) {
          workletRef.current.port.postMessage({
            frame: audioData,
            type: 'audio',
            micDuration: micDurationRef.current
          })
          setIsSpeaking(true)
        }
      }

      worker.onerror = (e) => console.error('Decoder worker error:', e)
      decoderWorkerRef.current = worker

      const sampleRate = audioContextRef.current?.sampleRate || 48000
      worker.postMessage({
        command: 'init',
        bufferLength: Math.round(960 * sampleRate / 24000),
        decoderSampleRate: 24000,
        outputBufferSampleRate: sampleRate,
        resampleQuality: 0
      })

      setTimeout(() => {
        const bosPage = createWarmupBosPageInline()
        worker.postMessage({ command: 'decode', pages: bosPage })
      }, 100)

      setTimeout(resolve, 500)
    })
  }, [])

  function createWarmupBosPageInline() {
    const opusHead = new Uint8Array([
      0x4F, 0x70, 0x75, 0x73, 0x48, 0x65, 0x61, 0x64,
      0x01, 0x01, 0x38, 0x01, 0x80, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00,
    ])
    const pageHeader = new Uint8Array([
      0x4F, 0x67, 0x67, 0x53, 0x00, 0x02,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x01, 0x13,
    ])
    const bosPage = new Uint8Array(pageHeader.length + opusHead.length)
    bosPage.set(pageHeader, 0)
    bosPage.set(opusHead, pageHeader.length)
    return bosPage
  }

  const initAudioContext = useCallback(async () => {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 })
    audioContextRef.current = audioContext

    const processorCode = `
      class MoshiProcessor extends AudioWorkletProcessor {
        constructor() {
          super()
          this.frames = []
          this.offsetInFirstBuffer = 0
          this.started = false
          this.remainingPartialBufferSamples = 0
          this.initialBufferSamples = Math.round(80 * sampleRate / 1000)
          this.partialBufferSamples = Math.round(10 * sampleRate / 1000)
          this.port.onmessage = (event) => {
            if (event.data.type === 'reset') {
              this.frames = []
              this.offsetInFirstBuffer = 0
              this.started = false
              return
            }
            this.frames.push(event.data.frame)
            if (this.currentSamples() >= this.initialBufferSamples && !this.started) {
              this.started = true
              this.remainingPartialBufferSamples = this.partialBufferSamples
            }
          }
        }
        currentSamples() {
          let samples = 0
          for (let i = 0; i < this.frames.length; i++) samples += this.frames[i].length
          return samples - this.offsetInFirstBuffer
        }
        canPlay() { return this.started && this.frames.length > 0 && this.remainingPartialBufferSamples <= 0 }
        process(inputs, outputs) {
          const output = outputs[0][0]
          this.remainingPartialBufferSamples -= output.length
          if (!this.canPlay()) return true
          let outIdx = 0
          while (outIdx < output.length && this.frames.length) {
            const first = this.frames[0]
            const toCopy = Math.min(first.length - this.offsetInFirstBuffer, output.length - outIdx)
            output.set(first.subarray(this.offsetInFirstBuffer, this.offsetInFirstBuffer + toCopy), outIdx)
            this.offsetInFirstBuffer += toCopy
            outIdx += toCopy
            if (this.offsetInFirstBuffer === first.length) { this.offsetInFirstBuffer = 0; this.frames.shift() }
          }
          if (outIdx < output.length) this.started = false
          return true
        }
      }
      registerProcessor('moshi-processor', MoshiProcessor)
    `

    const blob = new Blob([processorCode], { type: 'application/javascript' })
    const url = URL.createObjectURL(blob)
    await audioContext.audioWorklet.addModule(url)
    URL.revokeObjectURL(url)

    const workletNode = new AudioWorkletNode(audioContext, 'moshi-processor')
    workletNode.connect(audioContext.destination)
    workletRef.current = workletNode

    return audioContext
  }, [])

  const startRecording = useCallback(async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: 24000,
        channelCount: 1,
        echoCancellation: micSettings?.echoCancellation ?? true,
        noiseSuppression: micSettings?.noiseSuppression ?? true,
        autoGainControl: true
      }
    })
    mediaStreamRef.current = stream

    const source = audioContextRef.current.createMediaStreamSource(stream)
    const gainNode = audioContextRef.current.createGain()
    gainNode.gain.value = micSettings?.sensitivity ?? 1.0
    gainNodeRef.current = gainNode

    const analyser = audioContextRef.current.createAnalyser()
    analyser.fftSize = 256
    source.connect(gainNode)
    gainNode.connect(analyser)
    analyserRef.current = analyser

    const updateLevel = () => {
      if (!analyserRef.current) return
      const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
      analyserRef.current.getByteFrequencyData(dataArray)
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length
      setAudioLevel(average)
      animationFrameRef.current = requestAnimationFrame(updateLevel)
    }
    updateLevel()

    const audioContextSampleRate = audioContextRef.current.sampleRate
    const recorder = new Recorder({
      encoderPath: new URL('/assets/encoderWorker.min.js', import.meta.url).href,
      bufferLength: Math.round(960 * audioContextSampleRate / 24000),
      encoderFrameSize: 20,
      encoderSampleRate: 24000,
      maxFramesPerPage: 2,
      numberOfChannels: 1,
      recordingGain: micSettings?.sensitivity ?? 1,
      resampleQuality: 3,
      encoderComplexity: 0,
      encoderApplication: 2049,
      streamPages: true,
      mediaTrackConstraints: {
        echoCancellation: micSettings?.echoCancellation ?? true,
        noiseSuppression: micSettings?.noiseSuppression ?? true,
        autoGainControl: true
      }
    })

    recorder.ondataavailable = (data) => {
      micDurationRef.current = recorder.encodedSamplePosition / 48000
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        const message = encodeAudioMessage(data)
        socketRef.current.send(message)
      }
    }

    recorder.onstart = () => setStatus('recording')
    recorder.onstop = () => { if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current) }

    recorderRef.current = recorder
    recorder.start(stream)
  }, [micSettings])

  const stopRecording = useCallback(() => {
    if (recorderRef.current) { recorderRef.current.stop(); recorderRef.current = null }
    if (mediaStreamRef.current) { mediaStreamRef.current.getTracks().forEach(track => track.stop()); mediaStreamRef.current = null }
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current)
    setAudioLevel(0)
  }, [])

  useEffect(() => {
    let mounted = true

    async function setup() {
      await initAudioContext()
      await initDecoderWorker()
      if (!mounted) return

      const params = new URLSearchParams({ voice_prompt: `${voice}.pt`, text_prompt: persona })
      const wsUrl = `ws://${serverUrl}/api/chat?${params.toString()}`

      const socket = new WebSocket(wsUrl)
      socket.binaryType = 'arraybuffer'
      socketRef.current = socket

      socket.onopen = () => { if (mounted) setStatus('priming') }

      socket.onmessage = (event) => {
        if (!mounted) return
        const data = new Uint8Array(event.data)
        const message = decodeMessage(data)

        switch (message.type) {
          case 'handshake':
            setStatus('ready')
            startRecording()
            break
          case 'audio':
            if (decoderWorkerRef.current) {
              decoderWorkerRef.current.postMessage({ command: 'decode', pages: message.data }, [message.data.buffer])
            }
            break
          case 'text':
            currentTextRef.current += message.data
            setCurrentText(currentTextRef.current)
            if (currentTextRef.current.match(/[.!?]\s*$/)) {
              setTranscript(prev => [...prev, { speaker: 'ai', text: currentTextRef.current.trim(), timestamp: new Date() }])
              currentTextRef.current = ''
              setCurrentText('')
            }
            setIsSpeaking(true)
            break
          case 'error':
            setErrorMessage(message.data)
            setStatus('error')
            break
        }
      }

      socket.onclose = () => { if (mounted) { setStatus('connecting'); stopRecording() } }
      socket.onerror = (err) => { console.error('WebSocket error:', err); setErrorMessage('Connection error'); setStatus('error') }
    }

    setup()

    return () => {
      mounted = false
      stopRecording()
      if (socketRef.current) { socketRef.current.close(); socketRef.current = null }
      if (decoderWorkerRef.current) { decoderWorkerRef.current.terminate(); decoderWorkerRef.current = null }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') audioContextRef.current.close()
    }
  }, [voice, persona, serverUrl, micSettings, initAudioContext, initDecoderWorker, startRecording, stopRecording])

  const handleEnd = useCallback(async () => {
    stopRecording()
    if (socketRef.current) socketRef.current.close()

    if (transcript.length > 0) {
      const shouldSave = window.confirm('Save this conversation to history?')
      if (shouldSave) {
        try {
          await fetch(`http://${serverUrl}/api/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ voice_preset: voice, persona_prompt: persona, transcript })
          })
        } catch (err) { console.error('Failed to save:', err) }
      }
    }

    onEnd()
  }, [stopRecording, onEnd, transcript, serverUrl, voice, persona])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      {/* Status bar */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '0.5rem',
        marginBottom: '0.5rem',
        background: '#1e293b',
        borderRadius: '4px',
        fontSize: '0.7rem'
      }}>
        <span style={{ color: status === 'recording' ? '#22c55e' : status === 'error' ? '#ef4444' : '#eab308' }}>
          {status === 'connecting' ? 'Connecting...' :
           status === 'priming' ? 'Preparing...' :
           status === 'ready' ? 'Starting mic...' :
           status === 'recording' ? 'Listening' :
           status === 'error' ? errorMessage : status}
        </span>
        {isSpeaking && <span style={{ color: '#60a5fa' }}>Speaking</span>}
      </div>

      {/* Audio level */}
      {status === 'recording' && <AudioLevelMeter level={audioLevel} compact />}

      {/* Transcript */}
      <TranscriptList transcript={transcript} currentText={currentText} status={status} />
    </div>
  )
}

export default App
