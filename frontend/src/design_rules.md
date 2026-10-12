# Project Design Rules (Magnific Aesthetic Edition)

These design rules define the visual and architectural standards for the AutoGradeSystem Web Application, inspired by the clean, cinematic design language of Magnific.

---

## 1. Design Principles & Layout Composition
- **Copy the visual design language & layout principles, NOT product functionality.**
- **Reference A (Cinematic Presentation)**: Used for high-impact overview & pipeline spotlight banners:
  - Dark container backdrop (`#181113`) with subtle vignette overlays and high-contrast typography.
  - Large, bold headlines and clear hierarchy.
  - Pill badges and high-contrast action triggers (solid white and outlined pills).
- **Reference B (Application Dashboard)**:
  - Ultra-slim vertical icon sidebar on the left (`w-16` / `w-20`) with prominent accent action pill (`#F04F9B`).
  - Compact top header with breadcrumb trail (`AutoGrade System > ...`) and real-time environment status.
  - Generous card proportions, rounded corners (`rounded-2xl` and `rounded-3xl`), and subtle 1px border ring (`border-border`).
- **Reference C (Workspace & Editor)**:
  - Central two-column workspace for grading inputs and evaluation results.
  - Floating bottom capsule / control dock (`rounded-full`, glassmorphism backdrop-blur, subtle shadow).
  - Clear separation between inputs and model diagnostic results.

---

## 2. Strict Typography Rules
- **Font Family**: `Inter`, `-apple-system`, `BlinkMacSystemFont`, `sans-serif`. Monospace numbers use `JetBrains Mono`.
- **Exactly Two Font Weights**:
  - `font-normal` (400) — Body text, descriptions, table cells, secondary labels.
  - `font-semibold` (600) — Headings, titles, active tabs, buttons, metric numbers, and badges.
  - *No other font weights (300, 500, 700, 800) are permitted.*
- **Exactly Two General Text Colors**:
  - **Primary Text**: `#18181B` (`text-foreground` or `text-[#18181B]`).
  - **Muted Text**: `#71717A` (`text-muted-foreground` or `text-[#71717A]`).
  - *(Exceptions allowed only for functional semantic status badges and metric indicators).*

---

## 3. Color System & Design Tokens
- **Primary Background**: `#FFFFFF`
- **Secondary / Canvas Background**: `#F7F7F8`
- **Dark Hero / Accent Canvas**: `#181113`
- **Border**: `#E5E7EB` (`border-border`)
- **Signature Pink Accent**: `#F04F9B` (Magnific brand accent for hero pill badge, primary action highlight)
- **Tech Blue Accent**: `#5268F5` (Secondary highlights, export actions)
- **Status & Diagnostic Badges**:
  - Correct: `bg-emerald-50 text-emerald-700 border-emerald-200`
  - Partially Correct: `bg-amber-50 text-amber-700 border-amber-200`
  - Incorrect: `bg-rose-50 text-rose-700 border-rose-200`
  - Missing Concepts: `bg-sky-50 text-sky-700 border-sky-200`
  - Logical / Factual Error: `bg-red-50 text-red-700 border-red-200`
  - Grammar Error: `bg-orange-50 text-orange-700 border-orange-200`

---

## 4. Component Library Standard (shadcn/ui)
- Use standard shadcn/ui components (`Button`, `Card`, `Badge`, `Tabs`, `Table`, `Input`, `Textarea`, `Slider`, `Progress`, `Alert`, `Empty`, `Tooltip`, etc.) as first-class primitives.
- Prefer standard variants (`variant="outline"`, `variant="ghost"`, `variant="secondary"`) and compose rather than reinventing custom divs.
- Icon stroke width adapts to adjacent text:
  - Adjacent to `font-normal`: `strokeWidth={1.5}` or `strokeWidth={2}`.
  - Adjacent to `font-semibold`: `strokeWidth={2.5}`.

---

## 5. Global Consistency Over Local Optimizations
- Maintain identical spacing intervals (multiples of 4px / 8px: `gap-2`, `gap-3`, `gap-4`, `p-6`).
- Floating dock maintains fixed pill radius (`rounded-full`) across all workspace views.
