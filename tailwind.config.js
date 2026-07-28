/**
 * tailwind.config.js
 * -------------------------------------------------------------------
 * Design tokens for the terminal aesthetic.
 *
 * Colour discipline — the one rule that holds the whole UI together:
 *   AMBER  is interface chrome: labels, focus rings, active states.
 *   GREEN and RED are reserved EXCLUSIVELY for market direction.
 *
 * That means a red pixel always means "down" and never "delete" or
 * "error". Errors use amber. This is how real trading terminals work,
 * and it's why the signal colour reads instantly at a glance.
 * -------------------------------------------------------------------
 */

export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Base surfaces — deep blue-black, not neutral grey.
        void: '#070A0F',
        base: '#0A0E14',
        panel: '#121822',
        raised: '#18202C',
        hairline: '#1E2733',
        divider: '#2A3543',

        // Interface accent: terminal amber.
        amber: {
          DEFAULT: '#E8B339',
          dim: '#8A6B22',
          glow: 'rgba(232, 179, 57, 0.14)',
        },

        // Direction only.
        up: {
          DEFAULT: '#3DD68C',
          dim: '#1F6B47',
          glow: 'rgba(61, 214, 140, 0.12)',
        },
        down: {
          DEFAULT: '#FF5C5C',
          dim: '#7A2B2B',
          glow: 'rgba(255, 92, 92, 0.12)',
        },
        flat: {
          DEFAULT: '#8494AA',
          dim: '#3A4655',
        },

        // Text ramp.
        ink: '#E6EDF5',
        'ink-muted': '#8494AA',
        'ink-faint': '#5A6878',
      },

      fontFamily: {
        // Data is monospace. This is the workhorse face.
        mono: [
          'JetBrains Mono',
          'SF Mono',
          'Menlo',
          'Consolas',
          'Liberation Mono',
          'monospace',
        ],
        // Headings and prose.
        sans: ['Inter Tight', 'Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },

      fontSize: {
        // Micro-labels: uppercase, letterspaced, the terminal signature.
        micro: ['0.625rem', { lineHeight: '1rem', letterSpacing: '0.12em' }],
        tick: ['0.6875rem', { lineHeight: '1rem', letterSpacing: '0.04em' }],
        // Big price readouts.
        readout: ['1.75rem', { lineHeight: '2rem', letterSpacing: '-0.02em' }],
        hero: ['2.75rem', { lineHeight: '2.9rem', letterSpacing: '-0.03em' }],
      },

      borderRadius: {
        // Terminals are rectangular. Radius is near-zero throughout.
        panel: '2px',
        pill: '2px',
      },

      boxShadow: {
        panel: '0 1px 0 0 rgba(255,255,255,0.03) inset',
        'glow-up': '0 0 0 1px rgba(61, 214, 140, 0.35)',
        'glow-down': '0 0 0 1px rgba(255, 92, 92, 0.35)',
        'glow-amber': '0 0 0 1px rgba(232, 179, 57, 0.35)',
      },

      animation: {
        'pulse-tick': 'pulseTick 2.4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-up': 'fadeUp 0.32s ease-out both',
        'scan': 'scan 2.2s linear infinite',
      },

      keyframes: {
        pulseTick: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.25' },
        },
        fadeUp: {
          from: { opacity: '0', transform: 'translateY(6px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        scan: {
          from: { transform: 'translateX(-100%)' },
          to: { transform: 'translateX(100%)' },
        },
      },
    },
  },
  plugins: [],
};
