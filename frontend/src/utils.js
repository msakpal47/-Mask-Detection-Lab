export function drawBoxes(canvas, boxes) {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    boxes.forEach(box => {
        const { x, y, w, h, label, score } = box;
        
        let color = '#22c55e';
        if (label === 'Without Mask') color = '#ef4444';
        if (label === 'Improper Mask') color = '#f59e0b';
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, w, h);
        
        ctx.fillStyle = color;
        const text = `${label} ${Math.round(score * 100)}%`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(x, y - 25, textWidth + 10, 25);
        
        ctx.fillStyle = 'white';
        ctx.font = 'bold 14px sans-serif';
        ctx.fillText(text, x + 5, y - 7);
    });
}
