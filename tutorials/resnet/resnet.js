Ment.polluteGlobal();
let net = new Net([new FC(1, 5), new Emitter(5, 'a'), new Emitter(5, 'b'), new FC(5, 5), new Receiver('a'), new Receiver('b'), new FC(15, 1)]);
net.batchSize = 1;
for (var i = 0; i < 1000; i++) {
	net.train([0], [0]);
	net.train([1], [2]);
	net.train([2], [4]);
}
