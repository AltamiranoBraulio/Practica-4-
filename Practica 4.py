# Importación de librerías necesarias
import heapq  # Para usar colas de prioridad (implementación del min-heap)
import matplotlib.pyplot as plt  # Para visualización gráfica
import networkx as nx  # Para manipulación y análisis de grafos
import time  # Para pausas entre visualizaciones
import random  # Para generación de grafos aleatorios
import string  # Para obtener caracteres A-Z fácilmente

class PrimSimulator:
    def __init__(self, graph):
        """
        Constructor de la clase PrimSimulator.
        Inicializa el simulador con un grafo dado.
        
        Args:
            graph (dict): Grafo en formato de diccionario de adyacencia.
                         Ejemplo: {'A': [('B', 4), ('C', 2)], ...}
        """
        self.graph = graph  # Almacena el grafo original
        self.G = nx.Graph()  # Crea un grafo vacío de NetworkX para visualización
        
        # Construye el grafo de NetworkX a partir del grafo de entrada
        for node in graph:
            for neighbor, weight in graph[node]:
                self.G.add_edge(node, neighbor, weight=weight)  # Añade arista con peso
                
        # Calcula posiciones fijas para los nodos (layout del grafo)
        # seed=42 asegura que la disposición sea la misma en cada ejecución
        self.pos = nx.spring_layout(self.G, seed=42)

    def run_prim(self):
        """
        Ejecuta el algoritmo de Prim para encontrar el Árbol de Expansión Mínima (MST).
        
        Returns:
            list: Lista de aristas del MST en formato (nodo1, nodo2, peso)
        """
        mst = []  # Almacenará las aristas del MST resultante
        visited = set()  # Conjunto para nodos ya incluidos en el MST
        
        # Selecciona el primer nodo del grafo como punto de inicio
        start_node = list(self.graph.keys())[0]
        
        # Inicializa el heap (cola de prioridad) con las aristas del nodo inicial
        heap = [(weight, start_node, neighbor) for neighbor, weight in self.graph[start_node]]
        heapq.heapify(heap)  # Convierte la lista en un min-heap válido
        visited.add(start_node)  # Marca el nodo inicial como visitado
        
        # Configura la figura para visualización
        plt.figure(figsize=(14, 10))  # Tamaño grande para acomodar muchos nodos
        
        # Bucle principal del algoritmo de Prim
        while heap and len(visited) < len(self.graph):
            # Extrae la arista con menor peso del heap
            weight, u, v = heapq.heappop(heap)
            
            # Si el nodo destino no ha sido visitado
            if v not in visited:
                visited.add(v)  # Marca como visitado
                mst.append((u, v, weight))  # Añade la arista al MST
                print(f"Conectando {u} --{weight}--> {v}")  # Log de progreso
                
                # Actualiza la visualización
                self._draw_graph(mst, current_edge=(u, v))
                time.sleep(0.8)  # Pausa para visualización (más corta que en versión anterior)
                
                # Añade las aristas del nuevo nodo al heap
                for neighbor, w in self.graph[v]:
                    if neighbor not in visited:
                        heapq.heappush(heap, (w, v, neighbor))  # Inserta en el heap
                        
        return mst  # Retorna el MST completo

    def _draw_graph(self, mst_edges, current_edge=None):
        """
        Dibuja el grafo con el estado actual del MST.
        
        Args:
            mst_edges (list): Aristas del MST hasta el momento
            current_edge (tuple, optional): Arista actual siendo procesada. Defaults to None.
        """
        plt.clf()  # Limpia la figura anterior
        
        # Dibuja todos los nodos
        nx.draw_networkx_nodes(self.G, self.pos, node_size=800, node_color='lightblue')
        
        # Dibuja todas las aristas posibles (transparentes)
        nx.draw_networkx_edges(self.G, self.pos, alpha=0.1, width=1)
        
        # Dibuja las aristas del MST en rojo y más gruesas
        nx.draw_networkx_edges(
            self.G, 
            self.pos, 
            edgelist=[(u, v) for u, v, _ in mst_edges], 
            edge_color='red', 
            width=2.5
        )
        
        # Si hay una arista actual, la dibuja en verde y más gruesa
        if current_edge:
            nx.draw_networkx_edges(
                self.G, 
                self.pos, 
                edgelist=[current_edge], 
                edge_color='green', 
                width=4
            )
        
        # Dibuja las etiquetas de los nodos
        nx.draw_networkx_labels(self.G, self.pos, font_size=10)
        
        # Prepara y dibuja las etiquetas de peso de las aristas
        edge_labels = {(u, v): d['weight'] for u, v, d in self.G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            self.G, 
            self.pos, 
            edge_labels=edge_labels, 
            font_size=8
        )
        
        # Configuración final del gráfico
        plt.title("Simulador de Prim: Red óptima de conexiones entre ciudades (A-Z)", fontsize=14)
        plt.axis('off')  # Oculta los ejes
        plt.pause(0.05)  # Pausa breve para actualización (más rápida que antes)

def generate_random_graph():
    """
    Genera un grafo aleatorio con nodos de la A a la Z.
    
    Returns:
        dict: Grafo aleatorio en formato de diccionario de adyacencia.
    """
    nodes = list(string.ascii_uppercase)  # Crea lista de nodos A-Z
    graph = {node: [] for node in nodes}  # Inicializa grafo vacío
    
    # Genera conexiones aleatorias
    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:  # Evita duplicados y auto-conexiones
            if random.random() < 0.2:  # 20% de probabilidad de conexión
                weight = random.randint(1, 20)  # Peso aleatorio entre 1-20
                graph[u].append((v, weight))  # Añade arista en ambas direcciones
                graph[v].append((u, weight))
                
    return graph

# Bloque principal de ejecución
if __name__ == "__main__":
    # Mensaje inicial
    print("=== Simulador de Árbol Parcial Mínimo (Prim) ===")
    print("Objetivo: Conectar 26 ciudades (A-Z) con el menor costo.\n")
    
    # Generación del grafo aleatorio
    city_graph = generate_random_graph()
    
    # Creación y ejecución del simulador
    simulator = PrimSimulator(city_graph)
    mst = simulator.run_prim()
    
    # Resultados finales
    print("\n=== Red óptima de carreteras ===")
    for u, v, weight in mst:
        print(f"{u} -- {v} (Costo: {weight})")
    
    # Cálculo y muestra del costo total
    total_cost = sum(weight for _, _, weight in mst)
    print(f"\nCosto total mínimo: {total_cost}")
    
    # Mantiene la ventana gráfica abierta
    plt.show()