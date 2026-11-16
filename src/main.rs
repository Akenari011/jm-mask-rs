//! jm-mask-rs —— 二次元 NSFW 打码服务（Rust 版，替代 Python onnx_api.py）
//!
//! 接口与 Python 版完全兼容：
//!   POST /process/  (multipart file, query: output_format/style/block/expand/conf/
//!                    mask_classes/fallback/fallback_cover_ratio/min_boxes)
//! 返回打码后的图片，并带 X-Category / X-Cover / X-Fallback / X-Skip 响应头。
//!
//! 推理走 onnxruntime（load-dynamic，复用系统 libonnxruntime.so），
//! 图像处理（letterbox / 解码 / NMS / 马赛克 / 模糊）全部为纯 Rust 实现。

use axum::{
    body::Body,
    extract::{Multipart, Query, State},
    http::{HeaderValue, Response, StatusCode},
    routing::post,
    Router,
};
use image::{codecs::jpeg::JpegEncoder, DynamicImage, RgbImage};
use serde::Deserialize;
use std::sync::Arc;
use std::time::Instant;

const SEG_SIZE: usize = 1024;
const CLS_SIZE: usize = 224;
const PAD_VALUE: u8 = 114;

const SEG_NAMES: [&str; 4] = ["breast", "anus", "female_genital", "male_genital"];
const CLS_NAMES: [&str; 5] = ["drawing", "hentai", "normal", "porn", "sexy"];

struct Engine {
    seg: parking_lot::Mutex<ort::session::Session>,
    cls: parking_lot::Mutex<ort::session::Session>,
    seg_input: String,
    cls_input: String,
}

#[derive(Deserialize)]
struct Params {
    #[serde(default = "d_fmt")]
    output_format: String,
    #[serde(default = "d_style")]
    style: String,
    #[serde(default = "d_block")]
    block: i32,
    #[serde(default = "d_expand")]
    expand: i32,
    #[serde(default = "d_conf")]
    conf: f32,
    #[serde(default)]
    mask_classes: String,
    #[serde(default = "d_fallback")]
    fallback: String,
    #[serde(default)]
    fallback_cover_ratio: f32,
    #[serde(default = "d_min_boxes")]
    min_boxes: i32,
}

fn d_fmt() -> String { "jpg".into() }
fn d_style() -> String { "mosaic".into() }
fn d_block() -> i32 { 18 }
fn d_expand() -> i32 { 8 }
fn d_conf() -> f32 { 0.1 }
fn d_fallback() -> String { "mosaic".into() }
fn d_min_boxes() -> i32 { 1 }

/// 双线性缩放（对齐 OpenCV INTER_LINEAR 的半像素约定）——裸行切片实现
fn resize_bilinear(src: &RgbImage, dw: u32, dh: u32) -> RgbImage {
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    let (dw, dh) = (dw as usize, dh as usize);
    let mut out = RgbImage::new(dw as u32, dh as u32);
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return out;
    }
    let src_buf = src.as_raw();
    let dst_buf = out.as_mut();
    let rx = sw as f32 / dw as f32;
    let ry = sh as f32 / dh as f32;
    // 预计算 x 方向的索引与权重
    let mut xs: Vec<(usize, usize, f32)> = Vec::with_capacity(dw);
    for x in 0..dw {
        let fx = (x as f32 + 0.5) * rx - 0.5;
        let x0 = fx.floor();
        let wx = fx - x0;
        let x0c = x0.max(0.0).min((sw - 1) as f32) as usize;
        let x1c = (x0 + 1.0).max(0.0).min((sw - 1) as f32) as usize;
        xs.push((x0c * 3, x1c * 3, wx));
    }
    for y in 0..dh {
        let fy = (y as f32 + 0.5) * ry - 0.5;
        let y0 = fy.floor();
        let wy = fy - y0;
        let y0c = y0.max(0.0).min((sh - 1) as f32) as usize;
        let y1c = (y0 + 1.0).max(0.0).min((sh - 1) as f32) as usize;
        let r0 = &src_buf[y0c * sw * 3..(y0c + 1) * sw * 3];
        let r1 = &src_buf[y1c * sw * 3..(y1c + 1) * sw * 3];
        let out_row = &mut dst_buf[y * dw * 3..(y + 1) * dw * 3];
        for (x, &(i0, i1, wx)) in xs.iter().enumerate() {
            for c in 0..3 {
                let top = r0[i0 + c] as f32 * (1.0 - wx) + r0[i1 + c] as f32 * wx;
                let bot = r1[i0 + c] as f32 * (1.0 - wx) + r1[i1 + c] as f32 * wx;
                out_row[x * 3 + c] = (top * (1.0 - wy) + bot * wy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// 最近邻缩放（对齐 OpenCV INTER_NEAREST）——裸行切片实现
fn resize_nearest(src: &RgbImage, dw: u32, dh: u32) -> RgbImage {
    let (sw, sh) = (src.width() as usize, src.height() as usize);
    let (dw, dh) = (dw as usize, dh as usize);
    let mut out = RgbImage::new(dw as u32, dh as u32);
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return out;
    }
    let src_buf = src.as_raw();
    let dst_buf = out.as_mut();
    let rx = sw as f32 / dw as f32;
    let ry = sh as f32 / dh as f32;
    let xmap: Vec<usize> = (0..dw)
        .map(|x| ((x as f32 * rx).floor() as usize).min(sw - 1))
        .collect();
    for y in 0..dh {
        let sy = ((y as f32 * ry).floor() as usize).min(sh - 1);
        let srow = &src_buf[sy * sw * 3..(sy + 1) * sw * 3];
        let drow = &mut dst_buf[y * dw * 3..(y + 1) * dw * 3];
        for (x, &sx) in xmap.iter().enumerate() {
            drow[x * 3..x * 3 + 3].copy_from_slice(&srow[sx * 3..sx * 3 + 3]);
        }
    }
    out
}

/// letterbox 到 size×size，返回 (图, 缩放比, 左padding, 上padding)
fn letterbox(src: &RgbImage, size: usize) -> (RgbImage, f32, i32, i32) {
    let (w, h) = (src.width() as f32, src.height() as f32);
    let r = (size as f32 / w).min(size as f32 / h);
    let nw = (w * r).round() as u32;
    let nh = (h * r).round() as u32;
    let dw = (size as u32 - nw) / 2;
    let dh = (size as u32 - nh) / 2;
    let rs = resize_bilinear(src, nw.max(1), nh.max(1));
    let mut out = RgbImage::from_pixel(size as u32, size as u32, image::Rgb([PAD_VALUE; 3]));
    let rs_buf = rs.as_raw();
    let out_buf = out.as_mut();
    let (rw, rh) = (rs.width() as usize, rs.height() as usize);
    for y in 0..rh {
        let ty = y + dh as usize;
        if ty >= size {
            break;
        }
        let srow = &rs_buf[y * rw * 3..(y + 1) * rw * 3];
        let start = ty * size * 3 + dw as usize * 3;
        let end = start + rw * 3;
        if end <= out_buf.len() {
            out_buf[start..end].copy_from_slice(srow);
        }
    }
    (out, r, dw as i32, dh as i32)
}

/// RGB 图 → NCHW f32 /255（裸字节实现）
fn to_nchw(img: &RgbImage) -> Vec<f32> {
    let n = (img.width() * img.height()) as usize;
    let raw = img.as_raw();
    let mut data = vec![0f32; 3 * n];
    for i in 0..n {
        let p = &raw[i * 3..i * 3 + 3];
        data[i] = p[0] as f32 / 255.0;
        data[n + i] = p[1] as f32 / 255.0;
        data[2 * n + i] = p[2] as f32 / 255.0;
    }
    data
}

/// 分离式高斯模糊（sigma 由 ksize 按 OpenCV 规则推导）
fn gaussian_blur(img: &RgbImage, ksize: i32) -> RgbImage {
    let k = if ksize % 2 == 0 { ksize + 1 } else { ksize }.max(3);
    let radius = (k / 2) as i32;
    let sigma = 0.3 * (((k - 1) as f32) * 0.5 - 1.0) + 0.8;
    let mut kernel = vec![0f32; (radius * 2 + 1) as usize];
    let mut sum = 0f32;
    for i in -radius..=radius {
        let v = (-(i * i) as f32 / (2.0 * sigma * sigma)).exp();
        kernel[(i + radius) as usize] = v;
        sum += v;
    }
    for v in kernel.iter_mut() {
        *v /= sum;
    }
    let (w, h) = (img.width() as i32, img.height() as i32);
    let mut tmp = RgbImage::new(img.width(), img.height());
    // 水平
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0f32; 3];
            for (ki, i) in (-radius..=radius).enumerate() {
                let sx = (x + i).clamp(0, w - 1);
                let p = img.get_pixel(sx as u32, y as u32);
                for c in 0..3 {
                    acc[c] += p[c] as f32 * kernel[ki];
                }
            }
            tmp.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([
                    acc[0].round().clamp(0.0, 255.0) as u8,
                    acc[1].round().clamp(0.0, 255.0) as u8,
                    acc[2].round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    // 垂直
    let mut out = RgbImage::new(img.width(), img.height());
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0f32; 3];
            for (ki, i) in (-radius..=radius).enumerate() {
                let sy = (y + i).clamp(0, h - 1);
                let p = tmp.get_pixel(x as u32, sy as u32);
                for c in 0..3 {
                    acc[c] += p[c] as f32 * kernel[ki];
                }
            }
            out.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([
                    acc[0].round().clamp(0.0, 255.0) as u8,
                    acc[1].round().clamp(0.0, 255.0) as u8,
                    acc[2].round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    out
}

/// 对矩形区域做马赛克（先缩到 w/block × h/block，再最近邻放大）
fn mosaic_rect(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, block: i32) {
    if x2 <= x1 || y2 <= y1 {
        return;
    }
    let bw = x2 - x1;
    let bh = y2 - y1;
    let b = block.max(2) as u32;
    let sw = (bw / b).max(1);
    let sh = (bh / b).max(1);
    let mut roi = RgbImage::new(bw, bh);
    for y in 0..bh {
        for x in 0..bw {
            roi.put_pixel(x, y, *img.get_pixel(x1 + x, y1 + y));
        }
    }
    let small = resize_nearest(&roi, sw, sh);
    let back = resize_nearest(&small, bw, bh);
    for y in 0..bh {
        for x in 0..bw {
            img.put_pixel(x1 + x, y1 + y, *back.get_pixel(x, y));
        }
    }
}

fn blur_rect(img: &mut RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, ksize: i32) {
    if x2 <= x1 || y2 <= y1 {
        return;
    }
    let bw = x2 - x1;
    let bh = y2 - y1;
    let mut roi = RgbImage::new(bw, bh);
    for y in 0..bh {
        for x in 0..bw {
            roi.put_pixel(x, y, *img.get_pixel(x1 + x, y1 + y));
        }
    }
    let blurred = gaussian_blur(&roi, ksize);
    for y in 0..bh {
        for x in 0..bw {
            img.put_pixel(x1 + x, y1 + y, *blurred.get_pixel(x, y));
        }
    }
}

fn mosaic_whole(img: &mut RgbImage, block: i32) {
    let (w, h) = (img.width(), img.height());
    mosaic_rect(img, 0, 0, w, h, block);
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = (a[0] + a[2]).min(b[0] + b[2]);
    let y2 = (a[1] + a[3]).min(b[1] + b[3]);
    let iw = (x2 - x1).max(0.0);
    let ih = (y2 - y1).max(0.0);
    let inter = iw * ih;
    let ua = a[2] * a[3] + b[2] * b[3] - inter;
    if ua <= 0.0 { 0.0 } else { inter / ua }
}

#[derive(Clone)]
struct Det {
    class_id: usize,
    score: f32,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

fn classify(engine: &Engine, img: &RgbImage) -> String {
    let small = resize_bilinear(img, CLS_SIZE as u32, CLS_SIZE as u32);
    let data = to_nchw(&small);
    let mut session = engine.cls.lock();
    let tensor = match ort::value::Tensor::from_array((
        [1i64, 3, CLS_SIZE as i64, CLS_SIZE as i64],
        data,
    )) {
        Ok(t) => t,
        Err(_) => return "unknown".to_string(),
    };
    let outputs = match session.run(ort::inputs![engine.cls_input.as_str() => tensor]) {
        Ok(o) => o,
        Err(_) => return "unknown".to_string(),
    };
    let (_, probs) = match outputs[0].try_extract_tensor::<f32>() {
        Ok(v) => v,
        Err(_) => return "unknown".to_string(),
    };
    let mut best = 0usize;
    let mut best_v = f32::MIN;
    for (i, v) in probs.iter().enumerate().take(CLS_NAMES.len()) {
        if *v > best_v {
            best_v = *v;
            best = i;
        }
    }
    CLS_NAMES.get(best).unwrap_or(&"unknown").to_string()
}

struct SegResult {
    dets: Vec<Det>,
}

fn segment(engine: &Engine, img: &RgbImage, conf: f32) -> SegResult {
    let (lb, _r, _dw, _dh) = letterbox(img, SEG_SIZE);
    let data = to_nchw(&lb);
    let mut session = engine.seg.lock();
    let tensor = match ort::value::Tensor::from_array((
        [1i64, 3, SEG_SIZE as i64, SEG_SIZE as i64],
        data,
    )) {
        Ok(t) => t,
        Err(_) => return SegResult { dets: vec![] },
    };
    let outputs = match session.run(ort::inputs![engine.seg_input.as_str() => tensor]) {
        Ok(o) => o,
        Err(_) => return SegResult { dets: vec![] },
    };
    let (shape, data) = match outputs[0].try_extract_tensor::<f32>() {
        Ok(v) => v,
        Err(_) => return SegResult { dets: vec![] },
    };
    // output0: [1, 40, 21504] → 40 = 4(box) + 4(cls) + 32(mask)
    let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    if dims.len() != 3 {
        return SegResult { dets: vec![] };
    }
    let (ch, anchors) = (dims[1], dims[2]);
    if ch < 8 {
        return SegResult { dets: vec![] };
    }
    let nc = ch - 4 - 32;
    let mut dets: Vec<Det> = Vec::new();
    for a in 0..anchors {
        let mut best_v = f32::MIN;
        let mut best_c = 0usize;
        for c in 0..nc {
            let v = data[(4 + c) * anchors + a];
            if v > best_v {
                best_v = v;
                best_c = c;
            }
        }
        if best_v > conf {
            dets.push(Det {
                class_id: best_c,
                score: best_v,
                cx: data[a],
                cy: data[anchors + a],
                w: data[2 * anchors + a],
                h: data[3 * anchors + a],
            });
        }
    }
    dets.sort_by(|x, y| y.score.partial_cmp(&x.score).unwrap_or(std::cmp::Ordering::Equal));
    // NMS (IoU 0.45)
    let mut keep: Vec<Det> = Vec::new();
    for d in dets.into_iter() {
        let a = [d.cx - d.w / 2.0, d.cy - d.h / 2.0, d.w, d.h];
        let mut ok = true;
        for k in keep.iter() {
            let b = [k.cx - k.w / 2.0, k.cy - k.h / 2.0, k.w, k.h];
            if iou(&a, &b) > 0.45 {
                ok = false;
                break;
            }
        }
        if ok {
            keep.push(d);
        }
    }
    SegResult { dets: keep }
}

async fn handle(
    State(engine): State<Arc<Engine>>,
    Query(p): Query<Params>,
    mut multipart: Multipart,
) -> Response<Body> {
    let t0 = Instant::now();
    // 取文件
    let mut bytes: Option<Vec<u8>> = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        if field.name() == Some("file") {
            if let Ok(b) = field.bytes().await {
                bytes = Some(b.to_vec());
                break;
            }
        }
    }
    let Some(raw) = bytes else {
        return resp(StatusCode::BAD_REQUEST, vec![], "image/jpeg", &[]);
    };
    let Ok(dynimg) = image::load_from_memory(&raw) else {
        return resp(StatusCode::BAD_REQUEST, vec![], "image/jpeg", &[]);
    };
    let mut img: RgbImage = dynimg.to_rgb8();
    let (w, h) = (img.width() as i32, img.height() as i32);

    let t_cls = Instant::now();
    let (category, seg, ms_cls) = std::thread::scope(|sc| {
        let e1 = &engine;
        let e2 = &engine;
        let img1 = &img;
        let img2 = &img;
        let conf = p.conf;
        let h_cls = sc.spawn(move || classify(e1, img1));
        let h_seg = sc.spawn(move || segment(e2, img2, conf));
        let cat = h_cls.join().unwrap_or_else(|_| "unknown".to_string());
        let seg = h_seg.join().unwrap_or(SegResult { dets: vec![] });
        (cat, seg, t_cls.elapsed().as_millis())
    });
    let ms_seg = ms_cls;
    let sensitive = matches!(category.as_str(), "hentai" | "porn" | "sexy");
    let mask_set: Vec<String> = p
        .mask_classes
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // 反算回原图坐标
    let (lb_w, lb_h) = (SEG_SIZE as f32, SEG_SIZE as f32);
    let _ = (lb_w, lb_h);
    let (src_w, src_h) = (w as f32, h as f32);
    let r = (SEG_SIZE as f32 / src_w).min(SEG_SIZE as f32 / src_h);
    let nw = (src_w * r).round();
    let nh = (src_h * r).round();
    let dw = ((SEG_SIZE as f32 - nw) / 2.0).floor();
    let dh = ((SEG_SIZE as f32 - nh) / 2.0).floor();

    let t_post = Instant::now();
    let mut cover_px: f64 = 0.0;
    let mut n_keep: i32 = 0;
    for d in seg.dets.iter() {
        let name = SEG_NAMES.get(d.class_id).copied().unwrap_or("");
        if !mask_set.is_empty() && !mask_set.iter().any(|m| m == name) {
            continue;
        }
        n_keep += 1;
        let x1f = (d.cx - d.w / 2.0 - dw) / r;
        let y1f = (d.cy - d.h / 2.0 - dh) / r;
        let x2f = (d.cx + d.w / 2.0 - dw) / r;
        let y2f = (d.cy + d.h / 2.0 - dh) / r;
        let cw = (x2f.min(src_w) - x1f.max(0.0)).max(0.0) as f64;
        let chh = (y2f.min(src_h) - y1f.max(0.0)).max(0.0) as f64;
        cover_px += cw * chh;
        let x1 = (x1f as i32 - p.expand).max(0);
        let y1 = (y1f as i32 - p.expand).max(0);
        let x2 = (x2f as i32 + p.expand).min(w);
        let y2 = (y2f as i32 + p.expand).min(h);
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        if p.style == "blur" {
            blur_rect(&mut img, x1 as u32, y1 as u32, x2 as u32, y2 as u32, 35);
        } else {
            mosaic_rect(&mut img, x1 as u32, y1 as u32, x2 as u32, y2 as u32, p.block);
        }
    }

    let cover_ratio = if w > 0 && h > 0 {
        cover_px / (w as f64 * h as f64)
    } else {
        0.0
    };
    let need_fallback = sensitive
        && p.fallback != "off"
        && (n_keep < p.min_boxes
            || (p.fallback_cover_ratio > 0.0 && (cover_ratio as f32) < p.fallback_cover_ratio));

    let mut headers: Vec<(&'static str, String)> = vec![
        ("X-Category", category.clone()),
        ("X-Cover", format!("{:.3}", cover_ratio)),
        ("X-Fallback", if need_fallback { "1".into() } else { "0".into() }),
    ];

    if need_fallback {
        if p.fallback == "skip" {
            headers.push(("X-Skip", "1".into()));
            return resp(StatusCode::OK, vec![], "image/jpeg", &headers);
        }
        if p.fallback == "blur" {
            img = gaussian_blur(&img, 35);
        } else {
            mosaic_whole(&mut img, p.block);
        }
    }

    let ms_post = t_post.elapsed().as_millis();
    let t_enc = Instant::now();
    let is_png = matches!(p.output_format.as_str(), "png");
    let mut buf: Vec<u8> = Vec::new();
    let ok = if is_png {
        DynamicImage::ImageRgb8(img.clone())
            .write_to(&mut std::io::Cursor::new(&mut buf), image::ImageFormat::Png)
            .is_ok()
    } else {
        let mut enc = JpegEncoder::new_with_quality(&mut buf, 95);
        enc.encode_image(&img).is_ok()
    };
    if !ok {
        return resp(StatusCode::INTERNAL_SERVER_ERROR, vec![], "image/jpeg", &headers);
    }
    let ms_enc = t_enc.elapsed().as_millis();
    let ms = t0.elapsed().as_millis();
    eprintln!(
        "[mask-rs] cat={} cover={:.3} fb={} dets={} total={}ms (cls={} seg={} post={} enc={})",
        category, cover_ratio, need_fallback, n_keep, ms, ms_cls, ms_seg, ms_post, ms_enc
    );
    let ct = if is_png { "image/png" } else { "image/jpeg" };
    resp(StatusCode::OK, buf, ct, &headers)
}

fn resp(
    status: StatusCode,
    body: Vec<u8>,
    ct: &str,
    headers: &[(&'static str, String)],
) -> Response<Body> {
    let mut b = Response::builder().status(status).header("Content-Type", ct);
    for (k, v) in headers {
        if let Ok(hv) = HeaderValue::from_str(v) {
            b = b.header(*k, hv);
        }
    }
    b.body(Body::from(body)).unwrap_or_else(|_| Response::new(Body::empty()))
}

async fn health() -> &'static str {
    "ok"
}

/// 与 Python 版 FastAPI 的 /docs 对齐（插件用它探测服务是否可用）
async fn docs() -> axum::response::Html<&'static str> {
    axum::response::Html(
        "<!doctype html><meta charset=\"utf-8\"><title>jm-mask-rs</title>\
         <h1>jm-mask-rs</h1><p>POST /process/ (multipart file)</p>\
         <p>query: output_format, style, block, expand, conf, mask_classes, fallback, fallback_cover_ratio, min_boxes</p>",
    )
}

async fn openapi() -> axum::response::Json<serde_json::Value> {
    axum::response::Json(serde_json::json!({"openapi": "3.0.0", "info": {"title": "jm-mask-rs", "version": "0.1.0"}, "paths": {"/process/": {"post": {}}}}))
}

#[tokio::main]
async fn main() {
    let seg_path = std::env::var("MASK_SEG_MODEL")
        .unwrap_or_else(|_| "models/segmentation_model.onnx".to_string());
    let cls_path = std::env::var("MASK_CLS_MODEL")
        .unwrap_or_else(|_| "models/classification_model.onnx".to_string());
    let addr = std::env::var("MASK_ADDR").unwrap_or_else(|_| "127.0.0.1:8000".to_string());
    let threads: usize = std::env::var("MASK_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| (std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4) / 2).max(1));

    let dylib = std::env::var("MASK_ORT_DYLIB").unwrap_or_else(|_| {
        "/home/ubuntu/AutoHajimiMosaic/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.32.0".to_string()
    });
    let env = ort::init_from(&dylib)
        .expect("init onnxruntime dylib")
        .build()
        .expect("ort environment");
    // 关闭 CPU 内存池，让 RSS 平稳（大图推理后不保留峰值 arena）
    let no_arena: Option<Vec<(String, String)>> = if std::env::var("MASK_NO_ARENA").map(|v| v != "0").unwrap_or(true) {
        Some(vec![("CPUExecutionProvider.use_arena".to_string(), "0".to_string())])
    } else {
        None
    };
    let mk = |path: &str, th: usize| -> ort::session::Session {
        let mut b = ort::session::Session::builder(&env)
            .expect("ort builder")
            .with_intra_threads(th)
            .expect("threads");
        if let Some(opts) = &no_arena {
            let cpus: Vec<_> = env
                .devices()
                .filter(|d| d.hardware_device().ty() == ort::memory::DeviceType::CPU)
                .collect();
            if !cpus.is_empty() {
                b = b.with_devices(cpus, Some(opts)).expect("with_devices");
            }
        }
        b.commit_from_file(path).expect("load model")
    };
    let seg = mk(&seg_path, threads);
    let seg_input = seg.inputs()[0].name().to_string();
    let cls = mk(&cls_path, threads);
    let cls_input = cls.inputs()[0].name().to_string();

    let engine = Arc::new(Engine {
        seg: parking_lot::Mutex::new(seg),
        cls: parking_lot::Mutex::new(cls),
        seg_input,
        cls_input,
    });

    let app = Router::new()
        .route("/process/", post(handle))
        .route("/process", post(handle))
        .route("/health", axum::routing::get(health))
        .route("/docs", axum::routing::get(docs))
        .route("/openapi.json", axum::routing::get(openapi))
        .with_state(engine);

    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    eprintln!("[mask-rs] listening on {addr} (threads={threads})");
    axum::serve(listener, app).await.expect("serve");
}
