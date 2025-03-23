// script.js
const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');
const scoreElement = document.getElementById('score');

const CELL_SIZE = 20;
const COLORS = {
    background: '#000',
    snake: '#0f0',
    food: '#f00',
    text: '#fff'
};

class Snake {
    constructor() {
        this.reset();
    }

    reset() {
        this.body = [{ x: canvas.width / 2, y: canvas.height / 2 }];
        this.direction = this.randomDirection();
        this.grow = false;
        this.score = 0;
    }

    randomDirection() {
        const directions = [
            { x: 0, y: -1 },  // up
            { x: 0, y: 1 },   // down
            { x: -1, y: 0 },  // left
            { x: 1, y: 0 }    // right
        ];
        return directions[Math.floor(Math.random() * directions.length)];
    }

    move() {
        const head = { ...this.body[0] };
        head.x += this.direction.x * CELL_SIZE;
        head.y += this.direction.y * CELL_SIZE;

        // Toroidal movement
        if (head.x >= canvas.width) head.x = 0;
        if (head.x < 0) head.x = canvas.width - CELL_SIZE;
        if (head.y >= canvas.height) head.y = 0;
        if (head.y < 0) head.y = canvas.height - CELL_SIZE;

        // Check collision with self
        if (this.body.some(segment => segment.x === head.x && segment.y === head.y)) {
            this.reset();
            return false;
        }

        this.body.unshift(head);
        if (!this.grow) {
            this.body.pop();
        } else {
            this.grow = false;
            this.score++;
            scoreElement.textContent = this.score;
        }
        return true;
    }

    autoControl(food) {
        const head = this.body[0];
        const desiredDir = { x: 0, y: 0 };

        // Simple pathfinding logic
        if (food.x < head.x && this.direction.x !== 1) desiredDir.x = -1;
        else if (food.x > head.x && this.direction.x !== -1) desiredDir.x = 1;
        else if (food.y < head.y && this.direction.y !== 1) desiredDir.y = -1;
        else if (food.y > head.y && this.direction.y !== -1) desiredDir.y = 1;

        // Validate direction
        const newDir = { x: desiredDir.x || this.direction.x, y: desiredDir.y || this.direction.y };

        if (this.isDirectionSafe(newDir)) {
            this.direction = newDir;
        } else {
            this.findSafeDirection();
        }
    }

    isDirectionSafe(dir) {
        const nextPos = {
            x: this.body[0].x + dir.x * CELL_SIZE,
            y: this.body[0].y + dir.y * CELL_SIZE
        };
        return !this.body.some(segment => segment.x === nextPos.x && segment.y === nextPos.y);
    }

    findSafeDirection() {
        const possibleDirs = [
            { x: 0, y: -1 },  // up
            { x: 0, y: 1 },   // down
            { x: -1, y: 0 },  // left
            { x: 1, y: 0 }    // right
        ].filter(dir => !(dir.x === -this.direction.x && dir.y === -this.direction.y) && this.isDirectionSafe(dir));

        if (possibleDirs.length > 0) {
            this.direction = possibleDirs[Math.floor(Math.random() * possibleDirs.length)];
        }
    }
}

class Food {
    constructor(snakeBody) {
        this.position = this.randomPosition(snakeBody);
    }

    randomPosition(snakeBody) {
        while (true) {
            const x = Math.floor(Math.random() * (canvas.width / CELL_SIZE)) * CELL_SIZE;
            const y = Math.floor(Math.random() * (canvas.height / CELL_SIZE)) * CELL_SIZE;
            if (!snakeBody.some(segment => segment.x === x && segment.y === y)) {
                return { x, y };
            }
        }
    }
}

class Game {
    constructor() {
        this.snake = new Snake();
        this.food = new Food(this.snake.body);
        this.lastTime = 0;
        this.gameSpeed = 100;
    }

    start() {
        requestAnimationFrame(this.loop.bind(this));
    }

    loop(timestamp) {
        if (timestamp - this.lastTime > this.gameSpeed) {
            this.update();
            this.draw();
            this.lastTime = timestamp;
        }
        requestAnimationFrame(this.loop.bind(this));
    }

    update() {
        this.snake.autoControl(this. 