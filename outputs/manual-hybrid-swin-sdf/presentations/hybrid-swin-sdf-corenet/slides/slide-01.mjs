const COLORS = {
  bg: "#F7FAFC",
  ink: "#111827",
  muted: "#475569",
  line: "#334155",
  inputFill: "#E8F1FB",
  inputStroke: "#58708D",
  lane3d: "#EEF8F5",
  box3d: "#DFF1ED",
  stroke3d: "#3E6F75",
  lane2d: "#FFF6E9",
  box2d: "#FDEBD1",
  stroke2d: "#8A6943",
  fusionFill: "#EDE7FA",
  fusionStroke: "#665A82",
  headFill: "#FBEAEA",
  headStroke: "#8B5E63",
  transparent: "#00000000",
};

function box(slide, ctx, name, x, y, w, h, fill, stroke, radius = 18, lineWidth = 1.6) {
  return ctx.addShape(slide, {
    name,
    geometry: "roundRect",
    left: x,
    top: y,
    width: w,
    height: h,
    fill,
    line: ctx.line(stroke, lineWidth),
    borderRadius: radius,
  });
}

function label(slide, ctx, name, text, x, y, w, h, opts = {}) {
  return ctx.addText(slide, {
    name,
    text,
    left: x,
    top: y,
    width: w,
    height: h,
    fontSize: opts.size ?? 20,
    color: opts.color ?? COLORS.ink,
    bold: opts.bold ?? false,
    typeface: opts.face ?? ctx.fonts.body,
    align: opts.align ?? "center",
    valign: opts.valign ?? "middle",
    insets: opts.insets ?? { left: 6, right: 6, top: 4, bottom: 4 },
    fill: COLORS.transparent,
    line: ctx.line(COLORS.transparent, 0),
  });
}

function module(slide, ctx, spec) {
  const shape = box(slide, ctx, spec.name, spec.x, spec.y, spec.w, spec.h, spec.fill, spec.stroke, 16, 1.8);
  label(slide, ctx, `${spec.name}-title`, spec.title, spec.x + 8, spec.y + spec.titleY, spec.w - 16, spec.titleH, {
    size: spec.titleSize ?? 20,
    bold: true,
    color: COLORS.ink,
  });
  if (spec.body) {
    label(slide, ctx, `${spec.name}-body`, spec.body, spec.x + 10, spec.y + spec.bodyY, spec.w - 20, spec.bodyH, {
      size: spec.bodySize ?? 14,
      color: COLORS.muted,
      align: "center",
      valign: "top",
      insets: { left: 4, right: 4, top: 0, bottom: 0 },
    });
  }
  return shape;
}

function arrow(slide, ctx, from, to, opts = {}) {
  const connector = slide.shapes.connect(from, to, {
    kind: opts.kind ?? "straight",
    fromSide: opts.fromSide,
    toSide: opts.toSide,
    line: ctx.line(opts.color ?? COLORS.line, opts.width ?? 1.8),
    tail: { type: "triangle", width: 3, length: 3 },
  });
  connector.bringToFront();
  return connector;
}

export async function slide01(presentation, ctx) {
  const slide = presentation.slides.add();
  slide.background.fill = COLORS.bg;

  label(slide, ctx, "title", "灵息", 0, 26, ctx.W, 48, {
    size: 34,
    bold: true,
    color: COLORS.ink,
    face: "Microsoft YaHei",
  });
  label(
    slide,
    ctx,
    "subtitle",
    "Hybrid-Swin-SDF-CoreNet | Lung nodule CT segmentation | 3D global context + 2D multi-view texture",
    0,
    78,
    ctx.W,
    28,
    { size: 16, color: COLORS.muted },
  );

  const lane3d = box(slide, ctx, "lane-3d", 285, 150, 720, 255, COLORS.lane3d, COLORS.stroke3d, 24, 1.1);
  const lane2d = box(slide, ctx, "lane-2d", 285, 475, 720, 255, COLORS.lane2d, COLORS.stroke2d, 24, 1.1);
  lane3d.sendToBack();
  lane2d.sendToBack();
  label(slide, ctx, "lane-3d-label", "3D Global Context Branch", 322, 164, 250, 28, {
    size: 16,
    bold: true,
    color: COLORS.stroke3d,
    align: "left",
  });
  label(slide, ctx, "lane-2d-label", "2D Multi-view Texture Branch", 322, 489, 280, 28, {
    size: 16,
    bold: true,
    color: COLORS.stroke2d,
    align: "left",
  });

  const input = module(slide, ctx, {
    name: "input-patch",
    x: 48,
    y: 350,
    w: 220,
    h: 110,
    fill: COLORS.inputFill,
    stroke: COLORS.inputStroke,
    title: "Input 3D CT Patch",
    titleY: 28,
    titleH: 24,
    titleSize: 19,
    body: "[B, 1, 96, 96, 96]\nlung nodule crop",
    bodyY: 56,
    bodyH: 42,
    bodySize: 13.5,
  });
  label(slide, ctx, "shared-input", "shared input", 214, 331, 130, 22, {
    size: 13,
    color: COLORS.muted,
  });

  const swin = module(slide, ctx, {
    name: "swinunetr",
    x: 350,
    y: 215,
    w: 190,
    h: 95,
    fill: COLORS.box3d,
    stroke: COLORS.stroke3d,
    title: "3D SwinUNETR",
    titleY: 19,
    titleH: 25,
    body: "global 3D context\nfull-resolution feature",
    bodyY: 48,
    bodyH: 38,
    bodySize: 13,
  });
  const proj = module(slide, ctx, {
    name: "projection",
    x: 595,
    y: 215,
    w: 190,
    h: 95,
    fill: COLORS.box3d,
    stroke: COLORS.stroke3d,
    title: "Projection",
    titleY: 14,
    titleH: 24,
    body: "1×1 Conv3D\nInstanceNorm3D\nLeakyReLU → C_f",
    bodyY: 42,
    bodyH: 48,
    bodySize: 12.5,
  });
  const pos = module(slide, ctx, {
    name: "position-encoding",
    x: 840,
    y: 215,
    w: 160,
    h: 95,
    fill: COLORS.box3d,
    stroke: COLORS.stroke3d,
    title: "Absolute 3D\nPosition Encoding",
    titleY: 14,
    titleH: 40,
    titleSize: 16,
    body: "coords + spacing\nadded to F_3D",
    bodyY: 58,
    bodyH: 32,
    bodySize: 12.5,
  });

  const mv2d = module(slide, ctx, {
    name: "multiview-projector",
    x: 350,
    y: 555,
    w: 235,
    h: 112,
    fill: COLORS.box2d,
    stroke: COLORS.stroke2d,
    title: "MultiView2DProjector",
    titleY: 22,
    titleH: 24,
    body: "axial / coronal / sagittal\nshared 2D texture encoder",
    bodyY: 50,
    bodyH: 28,
    bodySize: 11.6,
  });
  for (const [text, x] of [["A", 382], ["C", 439], ["S", 496]]) {
    box(slide, ctx, `view-chip-${text}`, x, 639, 46, 20, "#FFFFFF", COLORS.stroke2d, 9, 1);
    label(slide, ctx, `view-chip-${text}-label`, text, x, 638, 46, 20, {
      size: 12,
      bold: true,
      color: COLORS.muted,
    });
  }

  const recon = module(slide, ctx, {
    name: "reconstruct-3d",
    x: 635,
    y: 555,
    w: 205,
    h: 112,
    fill: COLORS.box2d,
    stroke: COLORS.stroke2d,
    title: "Reconstruct to 3D",
    titleY: 24,
    titleH: 24,
    body: "view-aligned features\nF_2D in C_f channels",
    bodyY: 54,
    bodyH: 38,
    bodySize: 12.5,
  });

  const fusion = module(slide, ctx, {
    name: "gated-fusion",
    x: 1085,
    y: 362,
    w: 190,
    h: 132,
    fill: COLORS.fusionFill,
    stroke: COLORS.fusionStroke,
    title: "GatedFusion3D",
    titleY: 28,
    titleH: 26,
    titleSize: 19,
    body: "adaptive channel gate\nF_3D + F_2D\nfused feature",
    bodyY: 58,
    bodyH: 64,
    bodySize: 13.5,
  });

  const mask = module(slide, ctx, {
    name: "mask-head",
    x: 1360,
    y: 245,
    w: 195,
    h: 90,
    fill: COLORS.headFill,
    stroke: COLORS.headStroke,
    title: "mask_head",
    titleY: 22,
    titleH: 25,
    body: "mask_logits",
    bodyY: 52,
    bodyH: 25,
    bodySize: 14,
  });
  const sdf = module(slide, ctx, {
    name: "sdf-head",
    x: 1360,
    y: 410,
    w: 195,
    h: 90,
    fill: COLORS.headFill,
    stroke: COLORS.headStroke,
    title: "sdf_head",
    titleY: 16,
    titleH: 24,
    body: "SDF (tanh)\nrange [-1, 1]",
    bodyY: 45,
    bodyH: 38,
    bodySize: 13.5,
  });
  const core = module(slide, ctx, {
    name: "core-head",
    x: 1360,
    y: 575,
    w: 195,
    h: 90,
    fill: COLORS.headFill,
    stroke: COLORS.headStroke,
    title: "core_head",
    titleY: 22,
    titleH: 25,
    body: "core_logits",
    bodyY: 52,
    bodyH: 25,
    bodySize: 14,
  });
  label(slide, ctx, "heads-label", "Lightweight 3D Prediction Heads", 1288, 200, 315, 28, {
    size: 15,
    bold: true,
    color: COLORS.muted,
    align: "left",
  });

  arrow(slide, ctx, input, swin, { kind: "elbow", fromSide: "right", toSide: "left" });
  arrow(slide, ctx, input, mv2d, { kind: "elbow", fromSide: "right", toSide: "left" });
  arrow(slide, ctx, swin, proj, { fromSide: "right", toSide: "left" });
  arrow(slide, ctx, proj, pos, { fromSide: "right", toSide: "left" });
  arrow(slide, ctx, mv2d, recon, { fromSide: "right", toSide: "left" });
  arrow(slide, ctx, pos, fusion, { kind: "elbow", fromSide: "right", toSide: "left" });
  arrow(slide, ctx, recon, fusion, { kind: "elbow", fromSide: "right", toSide: "left" });
  arrow(slide, ctx, fusion, mask, { kind: "elbow", fromSide: "right", toSide: "left" });
  arrow(slide, ctx, fusion, sdf, { fromSide: "right", toSide: "left" });
  arrow(slide, ctx, fusion, core, { kind: "elbow", fromSide: "right", toSide: "left" });

  label(slide, ctx, "f3d-label", "F_3D", 1000, 236, 70, 24, {
    size: 14,
    bold: true,
    color: COLORS.stroke3d,
  });
  label(slide, ctx, "f2d-label", "F_2D", 988, 616, 70, 24, {
    size: 14,
    bold: true,
    color: COLORS.stroke2d,
  });

  return slide;
}
