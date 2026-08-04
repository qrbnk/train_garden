---
name: Botanical Tech
colors:
  surface: '#f9f9ff'
  surface-dim: '#d3daea'
  surface-bright: '#f9f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f0f3ff'
  surface-container: '#e7eefe'
  surface-container-high: '#e2e8f8'
  surface-container-highest: '#dce2f3'
  on-surface: '#151c27'
  on-surface-variant: '#3f4943'
  inverse-surface: '#2a313d'
  inverse-on-surface: '#ebf1ff'
  outline: '#6f7973'
  outline-variant: '#bec9c1'
  surface-tint: '#1a6b4f'
  primary: '#00432f'
  on-primary: '#ffffff'
  primary-container: '#005d42'
  on-primary-container: '#88d3b1'
  inverse-primary: '#8ad6b3'
  secondary: '#575e70'
  on-secondary: '#ffffff'
  secondary-container: '#d9dff5'
  on-secondary-container: '#5c6274'
  tertiary: '#00432c'
  on-tertiary: '#ffffff'
  tertiary-container: '#005d3f'
  on-tertiary-container: '#4bdca0'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#a6f2cf'
  primary-fixed-dim: '#8ad6b3'
  on-primary-fixed: '#002115'
  on-primary-fixed-variant: '#005139'
  secondary-fixed: '#dce2f7'
  secondary-fixed-dim: '#c0c6db'
  on-secondary-fixed: '#141b2b'
  on-secondary-fixed-variant: '#404758'
  tertiary-fixed: '#6ffbbe'
  tertiary-fixed-dim: '#4edea3'
  on-tertiary-fixed: '#002113'
  on-tertiary-fixed-variant: '#005236'
  background: '#f9f9ff'
  on-background: '#151c27'
  surface-variant: '#dce2f3'
  success-light: '#dcfce7'
  success-dark: '#166534'
  warning-amber: '#d97706'
  warning-light: '#fef3c7'
  info-blue: '#3b82f6'
  info-light: '#dbeafe'
  surface-gray: '#f9fafb'
  border-subtle: '#e5e7eb'
typography:
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
  headline-md:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '600'
    lineHeight: 24px
  section-header:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '700'
    lineHeight: 20px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 22px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 18px
  label-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 16px
    letterSpacing: 0.05em
  chat-bubble:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: '1.625'
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  gap-xs: 8px
  gap-md: 12px
  gap-lg: 16px
  padding-card: 16px
  padding-chat-h: 24px
  sidebar-default: 256px
  vault-panel-default: 420px
---

## Brand & Style

The design system embodies a **Professional / Modern** aesthetic infused with an organic, "garden-centric" narrative. It is designed to feel supportive, reliable, and growth-oriented, catering to career seekers in the sustainability and technology sectors. 

The visual language balances the precision of SaaS dashboards with soft, illustrative touches that humanize the job search experience. It utilizes high information density to provide utility while maintaining an airy, approachable feel through generous white space and a "garden" inspired color palette. The UI relies on a clean three-pane architecture to manage complex data without overwhelming the user.

## Colors

The palette is anchored by **Emerald 900**, a deep, forest-inspired green that signifies growth and stability. This is contrasted against a sophisticated **Gray 900** for high-priority text and primary UI foundations. 

Semantic colors are pulled from a "seasonal" spectrum: vibrant greens for success, warm ambers for attention-heavy items like sticky notes, and clear blues for informational resource chips. The background is kept intentionally neutral and bright to allow these organic accents to lead the user's eye to key actions and status changes.

## Typography

This design system uses **Inter** exclusively to ensure maximum legibility across data-dense layouts. The typographic hierarchy is strictly controlled to manage information density. 

- **Headings** use heavier weights (600-700) to anchor the three-pane layout.
- **Body Text** in chat interactions utilizes a relaxed line height (1.625) to improve readability during long coaching sessions.
- **Micro-copy** and status labels use a condensed, uppercase style with wider letter spacing to differentiate them from interactive labels and content.

## Layout & Spacing

The layout follows a **Fixed-Fluid-Fixed** 3-column model:
1. **Navigation Pane (Fixed):** A narrow rail for primary app-level navigation.
2. **Action Pane (Fluid):** The central workspace (Chat or List views) that expands to fill available space.
3. **Intelligence Pane (Fixed/Collapsible):** A right-side "Vault" for contextual metadata and documents.

We use a 4px baseline grid. Standard component spacing is set at 16px to maintain a balance between density and clarity. For mobile, the Intelligence Pane is hidden behind a drawer, and the three-column layout collapses into a single-column vertical stack with 16px side margins.

## Elevation & Depth

Hierarchy is established through **Tonal Layers** and extremely subtle **Ambient Shadows**. 

- **Level 0 (Background):** Solid `#f9fafb` page surface.
- **Level 1 (Panels):** Pure white surfaces with a 1px `#e5e7eb` border. No shadows.
- **Level 2 (Interactive Cards):** White surface with a 1px border. On hover, a subtle `0 2px 8px rgba(0,0,0,.06)` shadow is applied.
- **Level 3 (Sticky Notes/Overlays):** Distinct depth is reserved for floating elements like coaching notes, using a dual-shadow approach (12% opacity) to mimic a physical object placed on top of the digital "garden."

## Shapes

The shape language is **Rounded**, reflecting the organic "Garden" theme while maintaining professional structure.

- **Standard Containers:** 8px radius for buttons and navigation items.
- **Content Cards:** 12px radius to feel softer and more approachable.
- **Pills/Tags:** Full 99px radius (pill-shaped) to clearly distinguish them from actionable buttons.
- **Chat Bubbles:** Utilize asymmetrical rounding. User bubbles have a sharp 4px bottom-right corner; Bot bubbles have a sharp 4px bottom-left corner. This creates a directional "tail" without using complex SVG shapes.

## Components

### Buttons & Inputs
- **Primary CTA:** Solid Gray-900 background with white text and 8px rounding.
- **Secondary Action:** Ghost style with Gray-200 border, transitioning to Gray-50 on hover.
- **Input Fields:** 1px Gray-200 border, 8px rounding, using Inter 14px. Focus state uses a 2px primary Emerald-900 ring.

### Chat System
- **User Bubbles:** Gray-900 background, white text, asymmetrical rounding.
- **Bot Bubbles:** White background, 1px Gray-200 border, asymmetrical rounding.
- **Typing Indicator:** Three bouncing dots using Gray-400.

### Status & Feedback
- **Pills:** Used for job categories and status. Backgrounds use "Light" variants of semantic colors (e.g., Green-100) with "Dark" text (e.g., Green-800).
- **Progress Bars:** Thin 4px height, using Emerald-500 for the fill and Gray-200 for the track.

### Specialized Components
- **Sticky Notes:** Amber-100 background with a slightly darker amber top-border and heavy shadow. Used for career coaching tips.
- **Vault Cards:** Highly dense cards used in the right panel for document previews, featuring a 12px font size for metadata.