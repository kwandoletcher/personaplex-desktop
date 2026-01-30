import React, { useState, useEffect, useRef } from 'react'
import { 
  Mic, MicOff, Save, History, Settings, 
  Play, Square, User, Bot, Volume2 
} from 'lucide-react'

// Voice presets from the server
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

// Preset personas
const PRESET_PERSONAS = [
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

// WebSocket connection hook
function useWebSocket() {
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState(null)
  const ws = useRef(null)

  useEffect(() => {
    connect()
    return () => {
      if (ws.current) {
        ws.current.close()
      }
    }
  }, [])

  const connect = () => {
    try {
      ws.current = new WebSocket('ws://localhost:8998')
      
      ws.current.onopen = () => {
        setConnected(true)
        setError(null)
      }
      
      ws.current.onclose = () => {
        setConnected(false)
      }
      
      ws.current.onerror = (err) => {
        setError('Connection error')
        setConnected(false)
      }
    } catch (err) {
      setError(err.message)
    }
  }

  const send = (data) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(data))
    }
  }

  return { connected, error, send, ws: ws.current }
}

// Voice Selection Screen
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
              {voice.type} • {voice.gender}
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
          Quick Presets (click to apply):
        </h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {PRESET_PERSONAS.map(preset => (
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
function ActiveConversation({ voice, persona, onEnd, onSave, ws }) {
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState([])
  const [showSaveDialog, setShowSaveDialog] = useState(false)

  useEffect(() => {
    if (ws) {
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        
        if (data.type === 'transcript') {
          setTranscript(prev => [...prev, data])
        } else if (data.type === 'conversation_ended') {
          setShowSaveDialog(true)
        }
      }
    }
  }, [ws])

  const toggleListening = () => {
    setIsListening(!isListening)
    // In a full implementation, this would start/stop audio capture
  }

  const handleEnd = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'end_conversation' }))
    }
  }

  const handleSave = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'save_conversation' }))
    }
    setShowSaveDialog(false)
    onSave()
  }

  return (
    <div style={{ padding: '2rem', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <div style={{ marginBottom: '1.5rem', padding: '1rem', background: '#1e293b', borderRadius: '8px' }}>
        <div style={{ fontSize: '0.875rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
          Voice: {voice}
        </div>
        <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
          Persona: {persona.substring(0, 100)}...
        </div>
      </div>

      <div style={{ 
        flex: 1, 
        overflowY: 'auto', 
        marginBottom: '1.5rem',
        padding: '1rem',
        background: '#0f172a',
        borderRadius: '8px'
      }}>
        {transcript.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#64748b', padding: '2rem' }}>
            Click the microphone to start speaking...
          </div>
        ) : (
          transcript.map((msg, idx) => (
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
                justifyContent: 'center'
              }}>
                {msg.speaker === 'user' ? <User size={16} /> : <Bot size={16} />}
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '0.25rem' }}>
                  {msg.speaker === 'user' ? 'You' : 'PersonaPlex'}
                </div>
                <div>{msg.text}</div>
              </div>
            </div>
          ))
        )}
      </div>

      <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem' }}>
        <button
          onClick={toggleListening}
          style={{
            width: '72px',
            height: '72px',
            borderRadius: '50%',
            background: isListening ? '#dc2626' : '#2563eb',
            color: 'white',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          {isListening ? <MicOff size={32} /> : <Mic size={32} />}
        </button>
        
        <button
          onClick={handleEnd}
          style={{
            width: '72px',
            height: '72px',
            borderRadius: '50%',
            background: '#475569',
            color: 'white',
            border: 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <Square size={24} />
        </button>
      </div>

      {showSaveDialog && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0,0,0,0.8)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center'
        }}>
          <div style={{
            background: '#1e293b',
            padding: '2rem',
            borderRadius: '8px',
            maxWidth: '400px',
            textAlign: 'center'
          }}>
            <h3 style={{ marginBottom: '1rem' }}>Save Conversation?</h3>
            <p style={{ color: '#94a3b8', marginBottom: '1.5rem' }}>
              Would you like to save this conversation to your history?
            </p>
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
              <button
                onClick={() => { setShowSaveDialog(false); onEnd(); }}
                style={{
                  padding: '0.5rem 1.5rem',
                  background: '#475569',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                Don't Save
              </button>
              <button
                onClick={handleSave}
                style={{
                  padding: '0.5rem 1.5rem',
                  background: '#2563eb',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer'
                }}
              >
                <Save size={16} style={{ marginRight: '0.5rem', display: 'inline' }} />
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// History Screen
function HistoryScreen({ onBack }) {
  const [conversations, setConversations] = useState([])
  const { send } = useWebSocket()

  useEffect(() => {
    // Request history from server
    send({ type: 'get_history', limit: 50 })
  }, [])

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
          ← Back
        </button>
        <h2 style={{ fontSize: '1.5rem', color: '#60a5fa' }}>
          Conversation History
        </h2>
      </div>

      {conversations.length === 0 ? (
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
                borderRadius: '8px',
                cursor: 'pointer'
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

// Main App Component
function App() {
  const [screen, setScreen] = useState('home') // home, voice, persona, conversation, history
  const [selectedVoice, setSelectedVoice] = useState(null)
  const [persona, setPersona] = useState('')
  const { connected, error, send, ws } = useWebSocket()

  const startConversation = () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      send({
        type: 'start_conversation',
        voice: selectedVoice,
        persona: persona
      })
    }
    setScreen('conversation')
  }

  const endConversation = () => {
    setScreen('home')
    setSelectedVoice(null)
    setPersona('')
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <header style={{
        padding: '1rem 2rem',
        background: '#1e293b',
        borderBottom: '1px solid #334155',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <h1 style={{ fontSize: '1.25rem', color: '#60a5fa' }}>
          PersonaPlex Desktop
        </h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: connected ? '#22c55e' : '#ef4444'
          }} />
          <span style={{ fontSize: '0.875rem', color: '#94a3b8' }}>
            {connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </header>

      {/* Main Content */}
      <main style={{ flex: 1, overflow: 'auto' }}>
        {screen === 'home' && (
          <div style={{ 
            height: '100%', 
            display: 'flex', 
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem'
          }}>
            <h2 style={{ fontSize: '2rem', marginBottom: '1rem', color: '#e2e8f0' }}>
              Welcome to PersonaPlex
            </h2>
            <p style={{ color: '#94a3b8', marginBottom: '2rem', textAlign: 'center', maxWidth: '600px' }}>
              Have real-time voice conversations with AI. Choose from 16 different voices and 
              customize the persona to create unique conversational experiences.
            </p>
            
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', justifyContent: 'center' }}>
              <button
                onClick={() => setScreen('voice')}
                style={{
                  padding: '1rem 2rem',
                  background: '#2563eb',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontSize: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <Play size={20} />
                Start New Conversation
              </button>
              
              <button
                onClick={() => setScreen('history')}
                style={{
                  padding: '1rem 2rem',
                  background: '#334155',
                  color: 'white',
                  border: 'none',
                  borderRadius: '8px',
                  cursor: 'pointer',
                  fontSize: '1rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <History size={20} />
                View History
              </button>
            </div>

            {!connected && (
              <div style={{ 
                marginTop: '2rem', 
                padding: '1rem', 
                background: '#451a1a', 
                borderRadius: '8px',
                color: '#fca5a5'
              }}>
                ⚠️ Server not connected. Make sure the Python backend is running in WSL2.
              </div>
            )}
          </div>
        )}

        {screen === 'voice' && (
          <VoiceSelection
            selectedVoice={selectedVoice}
            setSelectedVoice={setSelectedVoice}
            onNext={() => setScreen('persona')}
          />
        )}

        {screen === 'persona' && (
          <PersonaEditor
            persona={persona}
            setPersona={setPersona}
            onBack={() => setScreen('voice')}
            onStart={startConversation}
          />
        )}

        {screen === 'conversation' && (
          <ActiveConversation
            voice={selectedVoice}
            persona={persona}
            onEnd={endConversation}
            onSave={endConversation}
            ws={ws}
          />
        )}

        {screen === 'history' && (
          <HistoryScreen onBack={() => setScreen('home')} />
        )}
      </main>
    </div>
  )
}

export default App
