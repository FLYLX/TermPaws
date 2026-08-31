import {
  createFileRoute,
  Outlet,
  redirect,
  useLocation,
} from "@tanstack/react-router"
import { useEffect, useState } from "react"

import { HangTagNav } from "@/components/Layout/HangTagNav"
import { isLoggedIn } from "@/hooks/useAuth"

export const Route = createFileRoute("/_layout")({
  component: Layout,
  beforeLoad: async () => {
    if (!isLoggedIn()) {
      throw redirect({
        to: "/login",
      })
    }
  },
})

const HANG_LAYOUT_CSS = `
.hang-layout { position: relative; min-height: 100svh; background: linear-gradient(135deg, #f5f1e8 0%, #f2f4ef 55%, #e9efec 100%); color: #1f1f1f; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.hang-layout::before { content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none; background-image: linear-gradient(to right, rgba(58,58,58,0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(58,58,58,0.05) 1px, transparent 1px); background-size: 28px 28px; }
.hang-layout > * { position: relative; z-index: 1; }
.tag-rail { position: relative; z-index: 35; display: flex; align-items: flex-start; justify-content: center; gap: 14px; padding: 14px 96px 8px; background: linear-gradient(180deg, #f7f5ef 0%, #f1efe9 55%, rgba(244,242,236,0) 100%); }
.tag-rail-line { position: absolute; top: 8px; left: 0; right: 0; height: 2px; background: #555; }
.tag-user-anchor { position: absolute; right: 20px; top: 36px; }
.hang-tag { position: relative; margin-top: 0; transform-origin: top center; transform: translateY(calc(-100% + 12px)); border: 2px solid #3a3a3a; border-top: none; background: #fff; color: #242424; padding: 18px 14px 9px; font-weight: 700; font-size: 12px; line-height: 1.2; display: flex; flex-direction: column; align-items: center; gap: 4px; box-shadow: 3px 4px 0 rgba(0,0,0,.10); border-radius: 0 0 18px 18px; cursor: pointer; transition: transform .3s cubic-bezier(.2,.8,.3,1), box-shadow .18s ease, background .18s ease; }
.hang-tag::before { content: ""; position: absolute; bottom: 100%; left: 50%; width: 2px; height: 60px; background: #555; }
.tag-hole { position: absolute; top: 5px; left: 50%; margin-left: -5px; width: 10px; height: 10px; border: 2px solid #3a3a3a; border-radius: 9999px; background: #f4f4f3; }
.hang-tag:hover { transform: translateY(calc(-50% + 6px)) rotate(0deg); background: #fbfbfa; box-shadow: 4px 6px 0 rgba(0,0,0,.14); }
.hang-tag-active, .hang-tag-active:hover { transform: translateY(0) rotate(0deg); background: #e9e9e6; box-shadow: 4px 7px 0 rgba(0,0,0,.18); }
.tag-user-menu { position: absolute; right: 0; top: calc(100% + 10px); z-index: 60; width: 220px; background: #fff; border: 2px solid #3a3a3a; border-radius: 12px; box-shadow: 4px 5px 0 rgba(0,0,0,.12); padding: 10px; animation: tag-pop .16s ease-out; }
@keyframes tag-pop { from { transform: translateY(-6px) scale(.96); opacity: 0; } to { transform: none; opacity: 1; } }
.sketch-mini-box { border: 2px solid #3a3a3a; border-radius: 10px; background: #fff; font-weight: 700; font-size: 12px; }
.hang-page { padding: 6px 18px 14px; animation: page-drop .3s cubic-bezier(.2,.8,.3,1); }
.full-bleed { margin: -22px -18px -40px; }
@keyframes page-drop { from { transform: translateY(-22px); opacity: 0; } to { transform: none; opacity: 1; } }
.hang-page .rounded-lg, .hang-page .rounded-xl, .hang-page .rounded-2xl, .hang-page .rounded-3xl, .hang-page .rounded-md { border-radius: 16px 205px 16px 205px / 205px 16px 205px 16px; }
.hang-page .rounded-full { border-radius: 9999px; }
.hang-page .border { border-color: #3d3d3d; }
.hang-page .border-t, .hang-page .border-b, .hang-page .border-l, .hang-page .border-r { border-color: #3d3d3d; }
.hang-page .shadow-sm { box-shadow: 2px 3px 0 rgba(0,0,0,.08); }
.hang-page .shadow, .hang-page .shadow-md { box-shadow: 3px 4px 0 rgba(0,0,0,.12); }
.hang-page .bg-card, .hang-page .bg-background, .hang-page .bg-accent\\/30 { background-color: #fff; }
.hang-page .bg-muted { background-color: #ececea; }
.hang-page .text-muted-foreground { color: #565654; }
.hang-page .text-foreground { color: #1f1f1f; }
.hang-page button, .hang-page [role="button"] { font-weight: 600; }
.hang-page input:not(.code-editor), .hang-page select { color: #1f1f1f; }
.hang-page textarea:not(.code-editor) { color: #1f1f1f; }
.version-badge { position: fixed; right: 12px; bottom: 10px; z-index: 10; border: 2px solid #3a3a3a; border-radius: 10px 3px 10px 3px / 3px 10px 3px 10px; background: rgba(255,255,255,.75); backdrop-filter: blur(2px); color: #565654; font-size: 11px; font-weight: 700; padding: 2px 8px; box-shadow: 2px 2px 0 rgba(0,0,0,.08); pointer-events: none; }
@media (max-width: 767px) {
  .tag-rail { padding: 14px 10px 8px; gap: 8px; }
  .hang-tag { padding: 16px 9px 7px; }
  .hang-tag > span:not(.tag-hole) { display: none; }
  .tag-user-anchor { right: 8px; }
  .tag-user-menu { right: 0; width: min(220px, calc(100vw - 20px)); }
  .hang-page { padding: 6px 8px 12px; }
  .full-bleed { margin: -14px -8px -28px; }
  .version-badge { right: 8px; bottom: 6px; }
  [data-sonner-toaster] { left: 10px !important; right: 10px !important; width: auto !important; }
}
`

function VersionBadge() {
  const [version, setVersion] = useState("")

  useEffect(() => {
    fetch("/api/v1/utils/version/")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data?.version) {
          setVersion(String(data.version))
        }
      })
      .catch(() => {})
  }, [])

  if (!version) {
    return null
  }
  return <div className="version-badge">v{version}</div>
}

function Layout() {
  const location = useLocation()

  return (
    <div className="hang-layout">
      <style>{HANG_LAYOUT_CSS}</style>
      <HangTagNav />
      <main key={location.pathname} className="hang-page">
        <Outlet />
      </main>
      <VersionBadge />
    </div>
  )
}

export default Layout
