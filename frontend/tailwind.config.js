/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#f5f7ff',
          100: '#ebf0ff',
          200: '#d1dcff',
          300: '#a3beff',
          400: '#7094ff',
          500: '#3d66ff',
          600: '#2943e6',
          700: '#1f2fb8',
          800: '#1a2594',
          900: '#172275',
          950: '#0f1445',
        }
      }
    },
  },
  plugins: [],
}

// ArchonAI UI components - architecture reference
